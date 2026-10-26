package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stream VByte delta-1 (scalar) — matches FastPFOR {@code svb_encode(..., delta=1, type=1)} wire format.
 */
public final class StreamVByteDelta1 {

    private StreamVByteDelta1() {}

    private static int encodedLen(int val) {
        long v = val & 0xFFFF_FFFFL;
        if (v < (1L << 8)) {
            return 1;
        }
        if (v < (1L << 16)) {
            return 2;
        }
        if (v < (1L << 24)) {
            return 3;
        }
        return 4;
    }

    private static int codeBits(int val) {
        long v = val & 0xFFFF_FFFFL;
        if (v < (1L << 8)) {
            return 0;
        }
        if (v < (1L << 16)) {
            return 1;
        }
        if (v < (1L << 24)) {
            return 2;
        }
        return 3;
    }

    private static int writeData(int val, byte[] out, int off) {
        long v = val & 0xFFFF_FFFFL;
        if (v < (1L << 8)) {
            out[off] = (byte) val;
            return 1;
        }
        if (v < (1L << 16)) {
            out[off] = (byte) val;
            out[off + 1] = (byte) (val >>> 8);
            return 2;
        }
        if (v < (1L << 24)) {
            out[off] = (byte) val;
            out[off + 1] = (byte) (val >>> 8);
            out[off + 2] = (byte) (val >>> 16);
            return 3;
        }
        out[off] = (byte) val;
        out[off + 1] = (byte) (val >>> 8);
        out[off + 2] = (byte) (val >>> 16);
        out[off + 3] = (byte) (val >>> 24);
        return 4;
    }

    private static int readData(byte[] in, int[] dataPtrRef, int code) {
        int p = dataPtrRef[0];
        int val;
        if (code == 0) {
            val = in[p] & 0xFF;
            p += 1;
        } else if (code == 1) {
            val = (in[p] & 0xFF) | ((in[p + 1] & 0xFF) << 8);
            p += 2;
        } else if (code == 2) {
            val = (in[p] & 0xFF) | ((in[p + 1] & 0xFF) << 8) | ((in[p + 2] & 0xFF) << 16);
            p += 3;
        } else {
            val = (in[p] & 0xFF)
                    | ((in[p + 1] & 0xFF) << 8)
                    | ((in[p + 2] & 0xFF) << 16)
                    | ((in[p + 3] & 0xFF) << 24);
            p += 4;
        }
        dataPtrRef[0] = p;
        return val;
    }

    static byte[] encode(int[] u32, int n) {
        int keyLen = (n + 3) / 4;
        int dataCap = 0;
        int prev = 0;
        for (int c = 0; c < n; c++) {
            int v = u32[c] - prev;
            prev = u32[c];
            dataCap += encodedLen(v);
        }
        byte[] out = new byte[4 + keyLen + dataCap];
        ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN).putInt(0, n);
        int keyPtr = 4;
        int dataPtr = 4 + keyLen;
        int shift = 0;
        int key = 0;
        prev = 0;
        for (int c = 0; c < n; c++) {
            if (shift == 8) {
                shift = 0;
                out[keyPtr++] = (byte) key;
                key = 0;
            }
            int val = u32[c] - prev;
            prev = u32[c];
            dataPtr += writeData(val, out, dataPtr);
            key |= codeBits(val) << shift;
            shift += 2;
        }
        out[keyPtr] = (byte) key;
        return Arrays.copyOf(out, dataPtr);
    }

    static void decode(byte[] buf, int[] outU32, int n) {
        int count = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt(0);
        if (count != n) {
            throw new IllegalArgumentException("count mismatch");
        }
        int keyLen = (n + 3) / 4;
        int keyPtr = 4;
        int[] dataPtrRef = {4 + keyLen};
        int shift = 0;
        int key = buf[keyPtr++] & 0xFF;
        int prev = 0;
        for (int c = 0; c < n; c++) {
            if (shift == 8) {
                shift = 0;
                key = buf[keyPtr++] & 0xFF;
            }
            int code = (key >> shift) & 3;
            int d = readData(buf, dataPtrRef, code);
            int val = d + prev;
            outU32[c] = val;
            prev = val;
            shift += 2;
        }
    }

    public static void benchStreamVByteDelta1Int(List<? extends List<Integer>> columns, long[] out) {
        benchStreamVByteDelta1Int(columns, out, null);
    }

    public static void benchStreamVByteDelta1Int(
            List<? extends List<Integer>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Integer> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int[] u32 = new int[n];
                for (int i = 0; i < n; i++) {
                    u32[i] = col.get(i);
                }
                long t0 = System.nanoTime();
                byte[] enc = encode(u32, n);
                long t1 = System.nanoTime();
                int[] dec = new int[n];
                long t2 = System.nanoTime();
                decode(enc, dec, n);
                long t3 = System.nanoTime();
                if (!Arrays.equals(u32, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Long> origList = new ArrayList<>(n);
                    List<Long> decodedList = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        origList.add((long) u32[i]);
                        decodedList.add((long) dec[i]);
                    }
                    BenchmarkLoss.accumulateLongListLoss(origList, decodedList, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += enc.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    public static void benchStreamVByteDelta1Long(List<? extends List<Long>> columns, long[] out) {
        benchStreamVByteDelta1Long(columns, out, null);
    }

    public static void benchStreamVByteDelta1Long(
            List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                orig += (long) n * BenchmarkWireFormat.FLOAT_BYTES;
                long[] data = new long[n];
                for (int i = 0; i < n; i++) {
                    Long v = col.get(i);
                    data[i] = v == null ? 0L : v;
                }
                long t0 = System.nanoTime();
                byte[] enc = Int64ColumnCodec.streamVByteEncodeInt64(data, n);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                long[] dec = Int64ColumnCodec.streamVByteDecodeInt64(enc, n);
                long t3 = System.nanoTime();
                if (!Arrays.equals(data, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Long> origList = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        origList.add(col.get(i));
                    }
                    List<Long> decList = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        decList.add(dec[i]);
                    }
                    BenchmarkLoss.accumulateLongListLoss(origList, decList, lossAcc);
                }
                comp += enc.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }
}
