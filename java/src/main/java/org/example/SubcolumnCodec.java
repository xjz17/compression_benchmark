package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Sub-column codec: scaling (for doubles) -> subcolumn -> bitpacking/RLE/dictionary. */
public final class SubcolumnCodec {
    private static final int BLOCK_SIZE = 512;

    private SubcolumnCodec() {}

    private static int bitWidth(long v) {
        return v == 0 ? 0 : 64 - Long.numberOfLeadingZeros(v);
    }

    private static final class BitWriter {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        int cur = 0;
        int used = 0;

        void write(long value, int bits) {
            for (int i = bits - 1; i >= 0; i--) {
                cur = (cur << 1) | (int) ((value >>> i) & 1L);
                used++;
                if (used == 8) {
                    out.write(cur);
                    cur = 0;
                    used = 0;
                }
            }
        }

        byte[] finish() {
            if (used > 0) {
                out.write(cur << (8 - used));
                cur = 0;
                used = 0;
            }
            return out.toByteArray();
        }
    }

    private static final class BitReader {
        final byte[] data;
        int pos = 0;

        BitReader(byte[] data) {
            this.data = data;
        }

        long read(int bits) {
            long out = 0;
            for (int i = 0; i < bits; i++) {
                int b = data[pos >>> 3] & 0xff;
                int bit = 7 - (pos & 7);
                out = (out << 1) | ((b >>> bit) & 1);
                pos++;
            }
            return out;
        }
    }

    private static byte[] pack(long[] vals, int n, int width) {
        if (width <= 0 || n == 0) {
            return new byte[0];
        }
        BitWriter bw = new BitWriter();
        for (int i = 0; i < n; i++) {
            bw.write(vals[i], width);
        }
        return bw.finish();
    }

    private static long[] unpack(byte[] data, int n, int width) {
        long[] out = new long[n];
        if (width <= 0) {
            return out;
        }
        BitReader br = new BitReader(data);
        for (int i = 0; i < n; i++) {
            out[i] = br.read(width);
        }
        return out;
    }

    private static byte[] withHeader(byte type, byte[] body) {
        byte[] out = new byte[1 + body.length];
        out[0] = type;
        System.arraycopy(body, 0, out, 1, body.length);
        return out;
    }

    private static void putBytes(ByteArrayOutputStream out, byte[] b) {
        out.write(b, 0, b.length);
    }

    private static void putU16(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
    }

    private static void putU32(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
        out.write((value >>> 16) & 0xff);
        out.write((value >>> 24) & 0xff);
    }

    private static void putI64(ByteArrayOutputStream out, long value) {
        for (int i = 0; i < 8; i++) {
            out.write((int) ((value >>> (8 * i)) & 0xff));
        }
    }

    private static byte[] encodeSubcolumn(long[] vals, int n, int beta) {
        long max = 0;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, vals[i]);
        }
        int width = bitWidth(max);
        byte[] bitBody = pack(vals, n, width);
        ByteArrayOutputStream bp = new ByteArrayOutputStream();
        bp.write(width);
        putU32(bp, bitBody.length);
        putBytes(bp, bitBody);
        byte bestType = 0;
        byte[] best = bp.toByteArray();

        long[] lensArr = new long[Math.max(1, n)];
        long[] valArr = new long[Math.max(1, n)];
        int runCount = 0;
        long maxLen = 0;
        if (n > 0) {
            long cur = vals[0];
            int start = 0;
            for (int i = 1; i < n; i++) {
                if (vals[i] != cur) {
                    int len = i - start;
                    lensArr[runCount] = len;
                    valArr[runCount] = cur;
                    maxLen = Math.max(maxLen, len);
                    runCount++;
                    cur = vals[i];
                    start = i;
                }
            }
            int len = n - start;
            lensArr[runCount] = len;
            valArr[runCount] = cur;
            maxLen = Math.max(maxLen, len);
            runCount++;
        }
        int lenWidth = bitWidth(maxLen);
        int valWidth = Math.max(width, 1);
        byte[] lenBytes = pack(lensArr, runCount, lenWidth);
        byte[] valBytes = pack(valArr, runCount, valWidth);
        ByteArrayOutputStream rle = new ByteArrayOutputStream();
        rle.write(lenWidth);
        rle.write(valWidth);
        putU16(rle, runCount);
        putU32(rle, lenBytes.length);
        putU32(rle, valBytes.length);
        putBytes(rle, lenBytes);
        putBytes(rle, valBytes);
        if (rle.size() < best.length) {
            bestType = 1;
            best = rle.toByteArray();
        }

        long[] keys = Arrays.copyOf(vals, n);
        Arrays.sort(keys);
        int keyCount = 0;
        for (int i = 0; i < n; i++) {
            if (i == 0 || keys[i] != keys[i - 1]) {
                keys[keyCount++] = keys[i];
            }
        }
        if (keyCount > 0 && keyCount <= Math.min(256, 1 << beta)) {
            long[] codes = new long[n];
            for (int i = 0; i < n; i++) {
                codes[i] = Arrays.binarySearch(keys, 0, keyCount, vals[i]);
            }
            int keyWidth = Math.max(width, 1);
            int codeWidth = bitWidth(keyCount - 1L);
            byte[] keyBytes = pack(keys, keyCount, keyWidth);
            byte[] codeBytes = pack(codes, n, codeWidth);
            ByteArrayOutputStream dict = new ByteArrayOutputStream();
            dict.write(keyWidth);
            dict.write(codeWidth);
            putU16(dict, keyCount);
            putU32(dict, keyBytes.length);
            putU32(dict, codeBytes.length);
            putBytes(dict, keyBytes);
            putBytes(dict, codeBytes);
            if (dict.size() < best.length) {
                bestType = 2;
                best = dict.toByteArray();
            }
        }
        return withHeader(bestType, best);
    }

    private static long[] decodeSubcolumn(ByteBuffer bb, int n) {
        int type = bb.get() & 0xff;
        if (type == 0) {
            int width = bb.get() & 0xff;
            int len = bb.getInt();
            byte[] body = new byte[len];
            bb.get(body);
            return unpack(body, n, width);
        }
        if (type == 1) {
            int lenWidth = bb.get() & 0xff;
            int valWidth = bb.get() & 0xff;
            int runs = bb.getShort() & 0xffff;
            int lenLen = bb.getInt();
            int valLen = bb.getInt();
            byte[] lenBytes = new byte[lenLen];
            bb.get(lenBytes);
            byte[] valBytes = new byte[valLen];
            bb.get(valBytes);
            long[] lens = unpack(lenBytes, runs, lenWidth);
            long[] vals = unpack(valBytes, runs, valWidth);
            long[] out = new long[n];
            int p = 0;
            for (int i = 0; i < runs; i++) {
                for (int j = 0; j < lens[i] && p < n; j++) {
                    out[p++] = vals[i];
                }
            }
            return out;
        }
        int keyWidth = bb.get() & 0xff;
        int codeWidth = bb.get() & 0xff;
        int keyCount = bb.getShort() & 0xffff;
        int keyLen = bb.getInt();
        int codeLen = bb.getInt();
        byte[] keyBytes = new byte[keyLen];
        bb.get(keyBytes);
        byte[] codeBytes = new byte[codeLen];
        bb.get(codeBytes);
        long[] keys = unpack(keyBytes, keyCount, keyWidth);
        long[] codes = unpack(codeBytes, n, codeWidth);
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = keys[(int) codes[i]];
        }
        return out;
    }

    public static byte[] encodeLongs(long[] values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        putU32(out, values.length);
        putU32(out, BLOCK_SIZE);
        for (int base = 0; base < values.length; base += BLOCK_SIZE) {
            int n = Math.min(BLOCK_SIZE, values.length - base);
            long min = Long.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                min = Math.min(min, values[base + i]);
            }
            long[] residual = new long[n];
            long max = 0;
            for (int i = 0; i < n; i++) {
                residual[i] = values[base + i] - min;
                if (Long.compareUnsigned(residual[i], max) > 0) {
                    max = residual[i];
                }
            }
            int maxBits = bitWidth(max);
            int beta = maxBits <= 1 ? 1 : 4;
            int groups = maxBits == 0 ? 0 : (maxBits + beta - 1) / beta;
            putI64(out, min);
            putU16(out, n);
            out.write(maxBits);
            out.write(beta);
            long mask = (1L << beta) - 1L;
            for (int g = 0; g < groups; g++) {
                long[] sub = new long[n];
                int shift = g * beta;
                for (int i = 0; i < n; i++) {
                    sub[i] = (residual[i] >>> shift) & mask;
                }
                byte[] enc = encodeSubcolumn(sub, n, beta);
                putU32(out, enc.length);
                putBytes(out, enc);
            }
        }
        return out.toByteArray();
    }

    public static long[] decodeLongs(byte[] blob) {
        ByteBuffer bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int total = bb.getInt();
        int blockSize = bb.getInt();
        if (blockSize <= 0) {
            throw new IllegalArgumentException("subcolumn block size");
        }
        long[] out = new long[total];
        int written = 0;
        while (written < total) {
            long min = bb.getLong();
            int n = bb.getShort() & 0xffff;
            int maxBits = bb.get() & 0xff;
            int beta = bb.get() & 0xff;
            int groups = maxBits == 0 ? 0 : (maxBits + beta - 1) / beta;
            long[] residual = new long[n];
            for (int g = 0; g < groups; g++) {
                int len = bb.getInt();
                int oldLimit = bb.limit();
                bb.limit(bb.position() + len);
                long[] sub = decodeSubcolumn(bb.slice().order(ByteOrder.LITTLE_ENDIAN), n);
                bb.position(bb.limit());
                bb.limit(oldLimit);
                int shift = g * beta;
                for (int i = 0; i < n; i++) {
                    residual[i] |= sub[i] << shift;
                }
            }
            for (int i = 0; i < n; i++) {
                out[written++] = min + residual[i];
            }
        }
        return out;
    }

    public static void benchSubcolumnLong(List<? extends List<Long>> columns, long[] result, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        for (List<Long> col : columns) {
            if (col.isEmpty()) {
                continue;
            }
            long[] data = new long[col.size()];
            for (int i = 0; i < col.size(); i++) {
                data[i] = col.get(i);
            }
            long t0 = System.nanoTime();
            byte[] enc = encodeLongs(data);
            long t1 = System.nanoTime();
            long[] dec = decodeLongs(enc);
            long t2 = System.nanoTime();
            if (!Arrays.equals(data, dec)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongArrayLoss(data, dec, data.length, lossAcc);
            }
            orig += BenchmarkWireFormat.columnOriginalSize(data.length, false);
            comp += enc.length;
            ct += t1 - t0;
            dt += t2 - t1;
        }
        result[0] = orig;
        result[1] = comp;
        result[2] = ct;
        result[3] = dt;
    }

    public static void benchSubcolumnDouble(
            List<? extends List<Double>> columns,
            List<Integer> maxPointPerColumn,
            long[] result,
            double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        for (int ci = 0; ci < columns.size(); ci++) {
            List<? extends Double> col = columns.get(ci);
            if (col.isEmpty()) {
                continue;
            }
            Integer cap = maxPointPerColumn != null && ci < maxPointPerColumn.size() ? maxPointPerColumn.get(ci) : null;
            int scale = -1;
            long[] data = null;
            if (cap != null && cap >= 0) {
                for (int s = Math.max(0, Math.min(cap, 24)); s >= 0; s--) {
                    long[] q = Int64ColumnCodec.quantizeDoubleColumn(col, s, false);
                    if (q != null) {
                        scale = s;
                        data = q;
                        break;
                    }
                }
            } else {
                scale = ExternalCompressionBench.maxLosslessDecimalScaleDoubleColumn(col);
                if (scale >= 0) {
                    data = Int64ColumnCodec.quantizeDoubleColumn(col, scale);
                }
            }
            if (scale < 0 || data == null) {
                continue;
            }
            long t0 = System.nanoTime();
            byte[] body = encodeLongs(data);
            ByteArrayOutputStream encOut = new ByteArrayOutputStream(4 + body.length);
            putU32(encOut, scale);
            putBytes(encOut, body);
            byte[] enc = encOut.toByteArray();
            long t1 = System.nanoTime();
            ByteBuffer bb = ByteBuffer.wrap(enc).order(ByteOrder.LITTLE_ENDIAN);
            int decScale = bb.getInt();
            byte[] bodyDec = new byte[bb.remaining()];
            bb.get(bodyDec);
            long[] dec = decodeLongs(bodyDec);
            long t2 = System.nanoTime();
            if (decScale != scale || !Arrays.equals(data, dec)) {
                continue;
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongArrayLoss(data, dec, data.length, lossAcc);
            }
            orig += BenchmarkWireFormat.columnOriginalSize(data.length, true);
            comp += enc.length;
            ct += t1 - t0;
            dt += t2 - t1;
        }
        result[0] = orig;
        result[1] = comp;
        result[2] = ct;
        result[3] = dt;
    }

}
