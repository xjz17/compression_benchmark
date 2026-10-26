package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Burtscher FPC v1.1 (DCC 2007) — Java port aligned with {@code cpp/src/fpc.c}.
 * Upstream reference: {@code benchmark/reference/upstream/fpc/fpc.c}.
 */
public final class FpcDouble {

    public static final String ROW_NAME = "fpc";
    public static final int DEFAULT_PRED_SIZE_LG2 = 16;

    private static final int SIZE = 32768;

    private static final long[] MASK = {
        0x0000000000000000L,
        0x00000000000000ffL,
        0x000000000000ffffL,
        0x0000000000ffffffL,
        0x000000ffffffffffL,
        0x0000ffffffffffffL,
        0x00ffffffffffffffL,
        0xffffffffffffffffL
    };

    private FpcDouble() {}

    /** Override default with {@code WEB_COMPRESSION_FPC_PRED_SIZE_LG2} (0..24). */
    public static int predSizeLg2ToBench() {
        String env = System.getenv("WEB_COMPRESSION_FPC_PRED_SIZE_LG2");
        if (env != null && !env.isEmpty()) {
            try {
                int v = Integer.parseInt(env.trim());
                if (v >= 0 && v <= 24) {
                    return v;
                }
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        return DEFAULT_PRED_SIZE_LG2;
    }

    private static long unsignedShiftRight(long v, int bits) {
        return v >>> bits;
    }

    private static long readU64LE(byte[] b, int off) {
        return (b[off] & 0xFFL)
                | ((b[off + 1] & 0xFFL) << 8)
                | ((b[off + 2] & 0xFFL) << 16)
                | ((b[off + 3] & 0xFFL) << 24)
                | ((b[off + 4] & 0xFFL) << 32)
                | ((b[off + 5] & 0xFFL) << 40)
                | ((b[off + 6] & 0xFFL) << 48)
                | ((b[off + 7] & 0xFFL) << 56);
    }

    private static void writeU64LE(byte[] b, int off, long v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
        b[off + 4] = (byte) (v >>> 32);
        b[off + 5] = (byte) (v >>> 40);
        b[off + 6] = (byte) (v >>> 48);
        b[off + 7] = (byte) (v >>> 56);
    }

    private static int bcodeFromXor(long xor1) {
        int bcode = 7;
        if (xor1 >> 56 == 0) {
            bcode = 6;
        }
        if (xor1 >> 48 == 0) {
            bcode = 5;
        }
        if (xor1 >> 40 == 0) {
            bcode = 4;
        }
        if (xor1 >> 24 == 0) {
            bcode = 3;
        }
        if (xor1 >> 16 == 0) {
            bcode = 2;
        }
        if (xor1 >> 8 == 0) {
            bcode = 1;
        }
        if (xor1 == 0) {
            bcode = 0;
        }
        return bcode;
    }

    private static void writeXorAtOffset(byte[] outbuf, int out, long xor1) {
        int base = (out >> 3) << 3;
        int bitShift = (out & 7) << 3;
        long lo = readU64LE(outbuf, base);
        lo |= xor1 << bitShift;
        writeU64LE(outbuf, base, lo);
        if (bitShift == 0) {
            xor1 = 0;
        }
        writeU64LE(outbuf, base + 8, unsignedShiftRight(xor1, 64 - bitShift));
    }

    private static long readXorAtOffset(byte[] inbuf, int in) {
        int base = (in >> 3) << 3;
        int tmp = (in & 7) << 3;
        long val = readU64LE(inbuf, base);
        long next = readU64LE(inbuf, base + 8);
        val = unsignedShiftRight(val, tmp);
        if (tmp == 0) {
            next = 0;
        } else {
            next = next << (64 - tmp);
        }
        return val | next;
    }

    private static int readChunk(byte[] in, long[] inbuf, int inBytes, int inCursor) {
        int remaining = (inBytes - inCursor) / 8;
        int read = Math.min(SIZE, remaining);
        for (int i = 0; i < read; i++) {
            inbuf[i] = readU64LE(in, inCursor + i * 8);
        }
        return read;
    }

    private static int streamRead(byte[] stream, byte[] buffer, int elemSize, int items, int cursor) {
        int remaining = (stream.length - cursor) / elemSize;
        int read = Math.min(items, remaining);
        System.arraycopy(stream, cursor, buffer, 0, read * elemSize);
        return read;
    }

    /**
     * Matches C {@code FPC_Compress_Memory}. Returns compressed bytes, or {@code null} when
     * {@code in.length} is not a multiple of 8 or compression would overflow the output buffer.
     */
    public static byte[] compressMemory(byte[] in, int predSizeLg2) {
        if (in == null || in.length == 0 || (in.length & 7) != 0) {
            return null;
        }
        final int inBytes = in.length;
        long[] inbuf = new long[SIZE + 1];
        byte[] outbuf = new byte[6 + (SIZE / 2) + (SIZE * 8) + 2];
        byte[] outStream = new byte[inBytes * 2 + 1024];
        int inCursor = 0;
        int outCursor = 0;

        outStream[outCursor++] = (byte) predSizeLg2;

        long predsizem1 = (1L << predSizeLg2) - 1;
        long hash = 0;
        long dhash = 0;
        long lastval = 0;
        long pred1 = 0;
        long pred2 = 0;
        long[] fcm = new long[(int) predsizem1 + 1];
        long[] dfcm = new long[(int) predsizem1 + 1];

        int intot = readChunk(in, inbuf, inBytes, inCursor);
        inCursor += intot * 8;
        while (intot > 0) {
            long val = inbuf[0];
            int out = 6 + ((intot + 1) >> 1);
            writeU64LE(outbuf, (out >> 3) << 3, 0);

            int bcode = 0;
            for (int i = 0; i < intot; i += 2) {
                long xor1 = val ^ pred1;
                fcm[(int) hash] = val;
                hash = ((hash << 6) ^ (unsignedShiftRight(val, 48))) & predsizem1;
                pred1 = fcm[(int) hash];

                long stride = val - lastval;
                long xor2 = val ^ (lastval + pred2);
                lastval = val;
                val = inbuf[i + 1];
                dfcm[(int) dhash] = stride;
                dhash = ((dhash << 2) ^ (unsignedShiftRight(stride, 40))) & predsizem1;
                pred2 = dfcm[(int) dhash];

                int code = 0;
                if (Long.compareUnsigned(xor1, xor2) > 0) {
                    code = 0x80;
                    xor1 = xor2;
                }
                bcode = bcodeFromXor(xor1);

                writeXorAtOffset(outbuf, out, xor1);
                out += bcode + (bcode >> 2);
                code |= bcode << 4;

                xor1 = val ^ pred1;
                fcm[(int) hash] = val;
                hash = ((hash << 6) ^ (unsignedShiftRight(val, 48))) & predsizem1;
                pred1 = fcm[(int) hash];

                stride = val - lastval;
                xor2 = val ^ (lastval + pred2);
                lastval = val;
                val = inbuf[i + 2];
                dfcm[(int) dhash] = stride;
                dhash = ((dhash << 2) ^ (unsignedShiftRight(stride, 40))) & predsizem1;
                pred2 = dfcm[(int) dhash];

                int bcode2 = code | 0x8;
                if (Long.compareUnsigned(xor1, xor2) > 0) {
                    code = bcode2;
                    xor1 = xor2;
                }
                bcode = bcodeFromXor(xor1);

                writeXorAtOffset(outbuf, out, xor1);
                out += bcode + (bcode >> 2);
                outbuf[6 + (i >> 1)] = (byte) (code | bcode);
            }
            if ((intot & 1) != 0) {
                out -= bcode + (bcode >> 2);
            }
            outbuf[0] = (byte) intot;
            outbuf[1] = (byte) (intot >> 8);
            outbuf[2] = (byte) (intot >> 16);
            outbuf[3] = (byte) out;
            outbuf[4] = (byte) (out >> 8);
            outbuf[5] = (byte) (out >> 16);

            if (outCursor + out > outStream.length) {
                return null;
            }
            System.arraycopy(outbuf, 0, outStream, outCursor, out);
            outCursor += out;

            intot = readChunk(in, inbuf, inBytes, inCursor);
            inCursor += intot * 8;
        }
        return Arrays.copyOf(outStream, outCursor);
    }

    /**
     * Matches C {@code FPC_Decompress_Memory}. Returns decompressed bytes, or {@code null} on failure.
     */
    public static byte[] decompressMemory(byte[] comp) {
        if (comp == null || comp.length < 7) {
            return null;
        }
        byte[] out = new byte[comp.length * 8 + SIZE * 8];
        int outLen = decompressMemory(comp, out, 0);
        if (outLen < 0) {
            return null;
        }
        return Arrays.copyOf(out, outLen);
    }

    private static int decompressMemory(byte[] comp, byte[] out, int outOff) {
        DecompressState st = decompressCore(comp, out, outOff);
        return st == null ? -1 : st.outLen;
    }

    /**
     * Decompress into {@code out}. Returns bytes consumed from {@code comp} (matching C return value),
     * or {@code -1} on failure.
     */
    public static int decompressMemory(byte[] comp, byte[] out) {
        if (comp == null || out == null || comp.length < 7) {
            return -1;
        }
        DecompressState st = decompressCore(comp, out, 0);
        return st == null ? -1 : st.consumed;
    }

    private static final class DecompressState {
        final int consumed;
        final int outLen;

        DecompressState(int consumed, int outLen) {
            this.consumed = consumed;
            this.outLen = outLen;
        }
    }

    private static DecompressState decompressCore(byte[] comp, byte[] out, int outOff) {
        if (comp == null || out == null || comp.length < 7) {
            return null;
        }
        byte[] inbuf = new byte[(SIZE / 2) + (SIZE * 8) + 6 + 2];
        long[] outbuf = new long[SIZE];
        int inCursor = 0;
        int outCursor = outOff;

        int ioc = streamRead(comp, inbuf, 1, 7, inCursor);
        inCursor += ioc;
        if (ioc == 1) {
            return null;
        }

        int predSizeLg2 = inbuf[0] & 0xFF;
        long predsizem1 = (1L << predSizeLg2) - 1;
        long hash = 0;
        long dhash = 0;
        long lastval = 0;
        long pred1 = 0;
        long pred2 = 0;
        long[] fcm = new long[(int) predsizem1 + 1];
        long[] dfcm = new long[(int) predsizem1 + 1];

        int intot = inbuf[3] & 0xFF;
        intot = (intot << 8) | (inbuf[2] & 0xFF);
        intot = (intot << 8) | (inbuf[1] & 0xFF);
        int in = inbuf[6] & 0xFF;
        in = (in << 8) | (inbuf[5] & 0xFF);
        in = (in << 8) | (inbuf[4] & 0xFF);

        do {
            int end = streamRead(comp, inbuf, 1, in, inCursor);
            inCursor += end;
            in = (intot + 1) >> 1;
            for (int i = 0; i < intot; i += 2) {
                int code = inbuf[i >> 1] & 0xFF;

                long val = readXorAtOffset(inbuf, in);
                int bcodeHi = (code >> 4) & 0x7;
                in += bcodeHi + (bcodeHi >> 2);
                val &= MASK[bcodeHi];

                if ((code & 0x80) != 0) {
                    pred1 = pred2;
                }
                val ^= pred1;

                fcm[(int) hash] = val;
                hash = ((hash << 6) ^ (unsignedShiftRight(val, 48))) & predsizem1;
                pred1 = fcm[(int) hash];

                long stride = val - lastval;
                dfcm[(int) dhash] = stride;
                dhash = ((dhash << 2) ^ (unsignedShiftRight(stride, 40))) & predsizem1;
                pred2 = val + dfcm[(int) dhash];
                lastval = val;
                outbuf[i] = val;

                val = readXorAtOffset(inbuf, in);
                int bcodeLo = code & 0x7;
                in += bcodeLo + (bcodeLo >> 2);
                val &= MASK[bcodeLo];

                if ((code & 0x8) != 0) {
                    pred1 = pred2;
                }
                val ^= pred1;

                fcm[(int) hash] = val;
                hash = ((hash << 6) ^ (unsignedShiftRight(val, 48))) & predsizem1;
                pred1 = fcm[(int) hash];

                stride = val - lastval;
                dfcm[(int) dhash] = stride;
                dhash = ((dhash << 2) ^ (unsignedShiftRight(stride, 40))) & predsizem1;
                pred2 = val + dfcm[(int) dhash];
                lastval = val;
                outbuf[i + 1] = val;
            }
            if (outCursor + intot * 8 > out.length) {
                return null;
            }
            for (int i = 0; i < intot; i++) {
                writeU64LE(out, outCursor + i * 8, outbuf[i]);
            }
            outCursor += intot * 8;

            intot = 0;
            if ((end - 6) >= in) {
                intot = inbuf[in + 2] & 0xFF;
                intot = (intot << 8) | (inbuf[in + 1] & 0xFF);
                intot = (intot << 8) | (inbuf[in] & 0xFF);
                end = inbuf[in + 5] & 0xFF;
                end = (end << 8) | (inbuf[in + 4] & 0xFF);
                end = (end << 8) | (inbuf[in + 3] & 0xFF);
                in = end;
            }
        } while (intot > 0);

        return new DecompressState(inCursor, outCursor - outOff);
    }

    public static void benchFpcDouble(List<? extends List<Double>> columns, long[] out) {
        benchFpcDouble(columns, out, null);
    }

    public static void benchFpcDouble(List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        int predSizeLg2 = predSizeLg2ToBench();
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
                if ((nbytes & 7) != 0) {
                    continue;
                }
                byte[] raw = new byte[nbytes];
                int p = 0;
                for (Double dv : col) {
                    long bits = Double.doubleToRawLongBits(dv == null ? Double.NaN : dv);
                    raw[p++] = (byte) bits;
                    raw[p++] = (byte) (bits >>> 8);
                    raw[p++] = (byte) (bits >>> 16);
                    raw[p++] = (byte) (bits >>> 24);
                    raw[p++] = (byte) (bits >>> 32);
                    raw[p++] = (byte) (bits >>> 40);
                    raw[p++] = (byte) (bits >>> 48);
                    raw[p++] = (byte) (bits >>> 56);
                }

                long t0 = System.nanoTime();
                byte[] compBytes = compressMemory(raw, predSizeLg2);
                long t1 = System.nanoTime();
                if (compBytes == null || compBytes.length == 0) {
                    continue;
                }

                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int consumed = decompressMemory(compBytes, dec);
                long t3 = System.nanoTime();
                if (consumed != compBytes.length || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Double> decVals = new ArrayList<>(n);
                    ByteBuffer db = ByteBuffer.wrap(dec).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < n; i++) {
                        decVals.add(db.getDouble());
                    }
                    List<Double> origCol = new ArrayList<>(col.size());
                    for (Double dv : col) {
                        origCol.add(dv);
                    }
                    BenchmarkLoss.accumulateDoubleListLoss(origCol, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compBytes.length;
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

    public static void benchFpcLong(List<? extends List<Long>> columns, long[] out) {
        benchFpcLong(columns, out, null);
    }

    public static void benchFpcLong(List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        int predSizeLg2 = predSizeLg2ToBench();
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int nbytes = n * 8;
                if ((nbytes & 7) != 0) {
                    continue;
                }
                byte[] raw = new byte[nbytes];
                int p = 0;
                for (Long lv : col) {
                    long v = lv == null ? 0L : lv;
                    raw[p++] = (byte) v;
                    raw[p++] = (byte) (v >>> 8);
                    raw[p++] = (byte) (v >>> 16);
                    raw[p++] = (byte) (v >>> 24);
                    raw[p++] = (byte) (v >>> 32);
                    raw[p++] = (byte) (v >>> 40);
                    raw[p++] = (byte) (v >>> 48);
                    raw[p++] = (byte) (v >>> 56);
                }

                long t0 = System.nanoTime();
                byte[] compBytes = compressMemory(raw, predSizeLg2);
                long t1 = System.nanoTime();
                if (compBytes == null || compBytes.length == 0) {
                    continue;
                }

                byte[] dec = new byte[nbytes];
                long t2 = System.nanoTime();
                int consumed = decompressMemory(compBytes, dec);
                long t3 = System.nanoTime();
                if (consumed != compBytes.length || !Arrays.equals(raw, dec)) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Long> decVals = new ArrayList<>(n);
                    ByteBuffer db = ByteBuffer.wrap(dec).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < n; i++) {
                        decVals.add(db.getLong());
                    }
                    List<Long> origCol = new ArrayList<>(col.size());
                    for (Long lv : col) {
                        origCol.add(lv);
                    }
                    BenchmarkLoss.accumulateLongListLoss(origCol, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compBytes.length;
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
