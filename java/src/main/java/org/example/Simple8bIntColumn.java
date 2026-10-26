package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Simple8b (Anh &amp; Moffat / Lemire FastPFOR tail-encoder order) on uint32 streams.
 *
 * <p>Rows: {@code simple8b_int} (int32 as unsigned LE) and {@code simple8b_float} (LeCo-style decimal-scaled
 * {@code int32} quantization, then the same Simple8b payload; wire prefixes LE {@code col_scale} for {@code 10^p}).
 *
 * <p>Wire: LE u32 {@code num_values}, LE u32 {@code num_u64_words}, then each compressed word as LE u64.
 */
public final class Simple8bIntColumn {

    private static final int LOGDESC = 4;
    private static final long MASK64 = 0xFFFFFFFFFFFFFFFFL;

    private static final class SelRow {
        final int maxNum;
        final int log;
        final int sel;

        SelRow(int maxNum, int log, int sel) {
            this.maxNum = maxNum;
            this.log = log;
            this.sel = sel;
        }

        int maxNum() {
            return maxNum;
        }

        int log() {
            return log;
        }

        int sel() {
            return sel;
        }
    }

    private static final SelRow[] SEL_ROWS = {
        new SelRow(240, 0, 0),
        new SelRow(120, 0, 1),
        new SelRow(60, 1, 2),
        new SelRow(30, 2, 3),
        new SelRow(20, 3, 4),
        new SelRow(15, 4, 5),
        new SelRow(12, 5, 6),
        new SelRow(10, 6, 7),
        new SelRow(8, 7, 8),
        new SelRow(7, 8, 9),
        new SelRow(6, 10, 10),
        new SelRow(5, 12, 11),
        new SelRow(4, 15, 12),
        new SelRow(3, 20, 13),
        new SelRow(2, 30, 14),
        new SelRow(1, 60, 15),
    };

    private Simple8bIntColumn() {}

    private static int which(long w) {
        return (int) ((w >>> 60) & 0xFL);
    }

    private static boolean tryMe(int[] buf, int base, int rem, int num, int log) {
        int c = Math.min(rem, num);
        if (log >= 32) {
            return true;
        }
        long lim = 1L << log;
        for (int i = 0; i < c; i++) {
            if (Integer.toUnsignedLong(buf[base + i]) >= lim) {
                return false;
            }
        }
        return true;
    }

    private static long composeWord(int sel, int[] vals, int from, int take, int log) {
        long w = sel & 0xFL;
        if (log == 0) {
            if (sel == 0) {
                return 0L;
            }
            if (sel == 1) {
                return 1L << 60;
            }
            throw new IllegalStateException("simple8b: compose log=0");
        }
        long mask = (1L << log) - 1L;
        if (log >= 32) {
            mask = 0xFFFFFFFFL;
        }
        for (int i = 0; i < take; i++) {
            long vv = Integer.toUnsignedLong(vals[from + i]) & mask;
            w = (w << log) | vv;
            w &= MASK64;
        }
        int totalBits = LOGDESC + log * take;
        int sh = 64 - totalBits;
        if (sh < 0 || sh > 64) {
            throw new IllegalStateException("simple8b: compose shift");
        }
        w = (w << sh) & MASK64;
        return w;
    }

    static long[] encodeU32(int[] vals, int n) {
        ArrayList<Long> out = new ArrayList<>();
        int pos = 0;
        while (pos < n) {
            int rem = n - pos;
            boolean wrote = false;
            for (SelRow row : SEL_ROWS) {
                if (!tryMe(vals, pos, rem, row.maxNum(), row.log())) {
                    continue;
                }
                int take = Math.min(rem, row.maxNum());
                long word;
                if (row.sel() <= 1) {
                    word = row.sel() == 0 ? 0L : (1L << 60);
                } else {
                    word = composeWord(row.sel(), vals, pos, take, row.log());
                }
                out.add(word);
                pos += take;
                wrote = true;
                break;
            }
            if (!wrote) {
                throw new IllegalStateException("simple8b: encode stuck");
            }
        }
        long[] arr = new long[out.size()];
        for (int i = 0; i < out.size(); i++) {
            arr[i] = out.get(i);
        }
        return arr;
    }

    private static SelRow rowForSel(int sel) {
        for (SelRow r : SEL_ROWS) {
            if (r.sel() == sel) {
                return r;
            }
        }
        return null;
    }

    static void decodeU32(long[] words, int nValues, int[] out) {
        int wi = 0;
        int op = 0;
        while (op < nValues) {
            if (wi >= words.length) {
                throw new IllegalArgumentException("simple8b: truncated stream");
            }
            long w = words[wi++] & MASK64;
            int sel = which(w);
            int rem = nValues - op;
            if (sel == 0) {
                int take = Math.min(rem, 240);
                for (int i = 0; i < take; i++) {
                    out[op++] = 0;
                }
            } else if (sel == 1) {
                int take = Math.min(rem, 120);
                for (int i = 0; i < take; i++) {
                    out[op++] = 0;
                }
            } else {
                SelRow row = rowForSel(sel);
                if (row == null || sel < 2) {
                    throw new IllegalArgumentException("simple8b: bad selector");
                }
                int maxnum = row.maxNum();
                int log = row.log();
                int take = Math.min(rem, maxnum);
                long mask = (1L << log) - 1L;
                if (log >= 32) {
                    mask = 0xFFFFFFFFL;
                }
                for (int k = 0; k < take; k++) {
                    int sh = 64 - LOGDESC - log - k * log;
                    out[op++] = (int) ((w >>> sh) & mask);
                }
            }
        }
        if (op != nValues) {
            throw new IllegalArgumentException("simple8b: length mismatch");
        }
    }

    /** @param raw little-endian int32 column bytes */
    public static byte[] encodeColumnBytes(byte[] raw) {
        if ((raw.length & 3) != 0) {
            throw new IllegalArgumentException("simple8b: length % 4");
        }
        int n = raw.length / 4;
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int[] u32 = new int[n];
        for (int i = 0; i < n; i++) {
            u32[i] = bb.getInt();
        }
        long[] words = encodeU32(u32, n);
        int nw = words.length;
        ByteBuffer out = ByteBuffer.allocate(8 + nw * 8).order(ByteOrder.LITTLE_ENDIAN);
        out.putInt(n);
        out.putInt(nw);
        for (int i = 0; i < nw; i++) {
            out.putLong(words[i]);
        }
        return out.array();
    }

    public static byte[] decodeColumnBytes(byte[] comp) {
        if (comp.length < 8) {
            throw new IllegalArgumentException("simple8b: truncated header");
        }
        ByteBuffer bb = ByteBuffer.wrap(comp).order(ByteOrder.LITTLE_ENDIAN);
        int n = bb.getInt();
        int nw = bb.getInt();
        int need = 8 + nw * 8;
        if (nw < 0 || comp.length < need) {
            throw new IllegalArgumentException("simple8b: truncated body");
        }
        long[] words = new long[nw];
        for (int i = 0; i < nw; i++) {
            words[i] = bb.getLong();
        }
        int[] u32 = new int[n];
        decodeU32(words, n, u32);
        ByteBuffer ob = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : u32) {
            ob.putInt(v);
        }
        return ob.array();
    }

    public static void benchSimple8bInt(List<? extends List<Integer>> columns, long[] result) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Integer> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                byte[] raw = new byte[n * 4];
                ByteBuffer wb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < n; i++) {
                    wb.putInt(col.get(i));
                }
                long t0 = System.nanoTime();
                byte[] enc = encodeColumnBytes(raw);
                long t1 = System.nanoTime();
                byte[] dec = decodeColumnBytes(enc);
                long t2 = System.nanoTime();
                if (!Arrays.equals(raw, dec)) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += enc.length;
                ct += t1 - t0;
                dt += t2 - t1;
            }
        } catch (Throwable ignored) {
            result[0] = 0;
            return;
        }
        result[0] = orig;
        result[1] = comp;
        result[2] = ct;
        result[3] = dt;
    }

    /**
     * Float columns: same lossless decimal scale + {@code int32} quantization as
     * {@link ExternalCompressionBench#benchLeCoFloatScaledIntegersFastPFor}, then Simple8b on int32 LE words.
     * Compressed bytes include a 4-byte LE {@code col_scale} prefix before the Simple8b block.
     */
    public static void benchSimple8bFloat(List<? extends List<Float>> columns, long[] result) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int colScale = 0;
                boolean scaleOk = true;
                for (int i = 0; i < n; i++) {
                    int s = ExternalCompressionBench.minDecimalScaleExactFloat(col.get(i));
                    if (s < 0) {
                        scaleOk = false;
                        break;
                    }
                    if (s > colScale) {
                        colScale = s;
                    }
                }
                if (!scaleOk) {
                    continue;
                }
                double pow = Math.pow(10.0, colScale);
                int[] data = new int[n];
                for (int i = 0; i < n; i++) {
                    float v = col.get(i);
                    long m = Math.round((double) v * pow);
                    if (m < Integer.MIN_VALUE || m > Integer.MAX_VALUE) {
                        data = null;
                        break;
                    }
                    int im = (int) m;
                    float r = (float) ((double) im / pow);
                    if (Float.floatToRawIntBits(r) != Float.floatToRawIntBits(v)) {
                        data = null;
                        break;
                    }
                    data[i] = im;
                }
                if (data == null) {
                    continue;
                }
                byte[] raw = new byte[n * 4];
                ByteBuffer wb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < n; i++) {
                    wb.putInt(data[i]);
                }
                long t0 = System.nanoTime();
                byte[] body = encodeColumnBytes(raw);
                byte[] enc =
                        ByteBuffer.allocate(4 + body.length)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .putInt(colScale)
                                .put(body)
                                .array();
                long t1 = System.nanoTime();
                ByteBuffer bb = ByteBuffer.wrap(enc).order(ByteOrder.LITTLE_ENDIAN);
                int sc = bb.getInt();
                byte[] bodyDec = new byte[bb.remaining()];
                bb.get(bodyDec);
                byte[] decRaw = decodeColumnBytes(bodyDec);
                long t2 = System.nanoTime();
                if (sc != colScale || !Arrays.equals(raw, decRaw)) {
                    continue;
                }
                ByteBuffer rb = ByteBuffer.wrap(decRaw).order(ByteOrder.LITTLE_ENDIAN);
                boolean bitsOk = true;
                for (int i = 0; i < n; i++) {
                    int im = rb.getInt();
                    float r = (float) ((double) im / pow);
                    if (Float.floatToRawIntBits(r) != Float.floatToRawIntBits(col.get(i))) {
                        bitsOk = false;
                        break;
                    }
                }
                if (!bitsOk) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += enc.length;
                ct += t1 - t0;
                dt += t2 - t1;
            }
        } catch (Throwable ignored) {
            result[0] = 0;
            return;
        }
        result[0] = orig;
        result[1] = comp;
        result[2] = ct;
        result[3] = dt;
    }

    /**
     * Double columns: lossless decimal scale on IEEE-754 doubles, quantize to {@code int64}, then Simple8b
     * on lo/hi int32 limbs plus 4-byte {@code col_scale} prefix.
     */
    public static void benchSimple8bDouble(List<? extends List<Double>> columns, long[] result) {
        benchSimple8bDouble(columns, null, result);
    }

    public static void benchSimple8bDouble(
            List<? extends List<Double>> columns, List<Integer> maxPointPerColumn, long[] result) {
        benchSimple8bDouble(columns, maxPointPerColumn, result, null);
    }

    public static void benchSimple8bDouble(
            List<? extends List<Double>> columns,
            List<Integer> maxPointPerColumn,
            long[] result,
            double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (int ci = 0; ci < columns.size(); ci++) {
                List<? extends Double> col = columns.get(ci);
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                Integer cap = maxPointPerColumn == null || ci >= maxPointPerColumn.size()
                        ? null
                        : maxPointPerColumn.get(ci);
                int colScale = -1;
                long[] data = null;
                boolean finite = true;
                for (Double dv : col) {
                    double v = dv == null ? Double.NaN : dv;
                    if (!Double.isFinite(v)) {
                        finite = false;
                        break;
                    }
                }
                if (cap != null && cap >= 0 && finite) {
                    for (int scale = Math.max(0, cap); scale >= 0; scale--) {
                        long[] candidate = Int64ColumnCodec.quantizeDoubleColumn(col, scale, false);
                        if (candidate != null) {
                            data = candidate;
                            colScale = scale;
                            break;
                        }
                    }
                }
                if (data == null) {
                    colScale = ExternalCompressionBench.maxLosslessDecimalScaleDoubleColumn(col);
                    if (colScale < 0) {
                        continue;
                    }
                    data = Int64ColumnCodec.quantizeDoubleColumn(col, colScale);
                    if (data == null) {
                        continue;
                    }
                }
                long t0 = System.nanoTime();
                byte[] body = Int64ColumnCodec.simple8bEncodeInt64(data, n);
                byte[] enc =
                        ByteBuffer.allocate(Integer.BYTES + body.length)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .putInt(colScale)
                                .put(body)
                                .array();
                long t1 = System.nanoTime();
                ByteBuffer bb = ByteBuffer.wrap(enc).order(ByteOrder.LITTLE_ENDIAN);
                int sc = bb.getInt();
                byte[] bodyDec = new byte[bb.remaining()];
                bb.get(bodyDec);
                long t2 = System.nanoTime();
                long[] recovered = Int64ColumnCodec.simple8bDecodeInt64(bodyDec, n);
                long t3 = System.nanoTime();
                if (sc != colScale || !Arrays.equals(data, recovered)) {
                    continue;
                }
                if (lossAcc != null) {
                    BenchmarkLoss.accumulateLongArrayLoss(data, recovered, n, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += enc.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            result[0] = 0;
            return;
        }
        result[0] = orig;
        result[1] = comp;
        result[2] = ct;
        result[3] = dt;
    }
}
