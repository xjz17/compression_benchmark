package org.example;

import com.github.luben.zstd.Zstd;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

/**
 * MOST-inspired lossy baseline: segment OLS + int16 residual quant + ZSTD.
 * Not the full mostdb codec (outlier table, slope mantissa truncation, shrinking cone).
 * Matches C++ {@code most_float::benchFloatColumnsMostMinimal} at rel error 1e-3.
 */
public final class MostFloatMinimal {

    private static final int K_SEG = 128;
    private static final float REL_ERROR = 1e-3f;
    private static final double REL_ERROR_DOUBLE = 1e-3;

    private MostFloatMinimal() {}

    private static void olsAb(float[] v, int m, float[] ab) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < m; i++) {
            double x = i;
            double y = v[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        double dm = m;
        double den = dm * sxx - sx * sx;
        if (Math.abs(den) < 1e-30) {
            ab[0] = m > 0 ? v[0] : 0f;
            ab[1] = 0f;
            return;
        }
        ab[1] = (float) ((dm * sxy - sx * sy) / den);
        ab[0] = (float) ((sy - ab[1] * sx) / dm);
    }

    private static byte[] packColumn(List<? extends Float> col, int n) {
        float colMax = 0f;
        for (int i = 0; i < n; i++) {
            colMax = Math.max(colMax, Math.abs(col.get(i)));
        }
        if (colMax == 0f) {
            colMax = 1f;
        }

        int est = n * 2 + (n / K_SEG + 2) * 32;
        ByteBuffer buf = ByteBuffer.allocate(est).order(ByteOrder.LITTLE_ENDIAN);
        float[] seg = new float[K_SEG];
        for (int off = 0; off < n; off += K_SEG) {
            int m = Math.min(K_SEG, n - off);
            for (int i = 0; i < m; i++) {
                seg[i] = col.get(off + i);
            }
            float[] ab = new float[2];
            olsAb(seg, m, ab);
            float resMax = 0f;
            float[] res = new float[m];
            for (int i = 0; i < m; i++) {
                float pred = ab[0] + ab[1] * (float) i;
                res[i] = seg[i] - pred;
                resMax = Math.max(resMax, Math.abs(res[i]));
            }
            float scale = resMax > 0f ? resMax * REL_ERROR : 1f;
            float inv = 1f / scale;
            buf.putInt(m);
            buf.putFloat(ab[0]);
            buf.putFloat(ab[1]);
            buf.putFloat(scale);
            for (int i = 0; i < m; i++) {
                long qd = Math.round((double) res[i] * (double) inv);
                if (qd > 32767) {
                    qd = 32767;
                }
                if (qd < -32768) {
                    qd = -32768;
                }
                buf.putShort((short) qd);
            }
        }
        return Arrays.copyOf(buf.array(), buf.position());
    }

    private static boolean verifyUnpack(List<? extends Float> col, int n, byte[] raw) {
        float colMax = 0f;
        for (int i = 0; i < n; i++) {
            colMax = Math.max(colMax, Math.abs(col.get(i)));
        }
        if (colMax == 0f) {
            colMax = 1f;
        }

        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (buf.remaining() >= 16) {
            int m = buf.getInt();
            float a = buf.getFloat();
            float b = buf.getFloat();
            float scale = buf.getFloat();
            if (m > K_SEG || off + m > n || buf.remaining() < m * 2) {
                return false;
            }
            for (int i = 0; i < m; i++) {
                short q = buf.getShort();
                float dec = a + b * (float) i + (float) q * scale;
                float orig = col.get(off + i);
                float err = Math.abs(dec - orig);
                float boundAbs = REL_ERROR * Math.max(Math.abs(orig), colMax);
                if (err > boundAbs * 1.0001f + 1e-6f) {
                    return false;
                }
            }
            off += m;
        }
        return off == n && !buf.hasRemaining();
    }

    private static void accumulateUnpackDoubleLoss(
            List<? extends Double> col, int n, byte[] raw, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return;
        }
        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (buf.remaining() >= 28) {
            int m = buf.getInt();
            double a = buf.getDouble();
            double b = buf.getDouble();
            double scale = buf.getDouble();
            if (m > K_SEG || off + m > n || buf.remaining() < m * 2) {
                return;
            }
            for (int i = 0; i < m; i++) {
                short q = buf.getShort();
                double dec = a + b * i + q * scale;
                Double origObj = col.get(off + i);
                double orig = origObj == null ? Double.NaN : origObj;
                if (Double.isFinite(orig) && Double.isFinite(dec)) {
                    lossAcc[0] += Math.abs(orig - dec);
                    lossAcc[1] += Math.abs(orig);
                }
            }
            off += m;
        }
    }

    private static void accumulateUnpackFloatLoss(
            List<? extends Float> col, int n, byte[] raw, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return;
        }
        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (buf.remaining() >= 16) {
            int m = buf.getInt();
            float a = buf.getFloat();
            float b = buf.getFloat();
            float scale = buf.getFloat();
            if (m > K_SEG || off + m > n || buf.remaining() < m * 2) {
                return;
            }
            for (int i = 0; i < m; i++) {
                short q = buf.getShort();
                float dec = a + b * (float) i + q * scale;
                float orig = col.get(off + i);
                if (Float.isFinite(orig) && Float.isFinite(dec)) {
                    lossAcc[0] += Math.abs((double) orig - (double) dec);
                    lossAcc[1] += Math.abs((double) orig);
                }
            }
            off += m;
        }
    }

    public static void benchMostFloatMinimal(List<? extends List<Float>> columns, long[] out) {
        benchMostFloatMinimal(columns, out, null);
    }

    public static void benchMostFloatMinimal(
            List<? extends List<Float>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                byte[] raw = packColumn(col, n);
                if (!verifyUnpack(col, n, raw)) {
                    continue;
                }
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t2 = System.nanoTime();
                if (d.length != raw.length || !Arrays.equals(raw, d)) {
                    continue;
                }
                if (!verifyUnpack(col, n, d)) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateUnpackFloatLoss(col, n, d, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += c.length;
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

    private static void olsAbDouble(double[] v, int m, double[] ab) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < m; i++) {
            double x = i;
            double y = v[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        double dm = m;
        double den = dm * sxx - sx * sx;
        if (Math.abs(den) < 1e-30) {
            ab[0] = m > 0 ? v[0] : 0.0;
            ab[1] = 0.0;
            return;
        }
        ab[1] = (dm * sxy - sx * sy) / den;
        ab[0] = (sy - ab[1] * sx) / dm;
    }

    private static byte[] packColumnDouble(List<? extends Double> col, int n) {
        double colMax = 0.0;
        for (int i = 0; i < n; i++) {
            Double v = col.get(i);
            double dv = v == null ? Double.NaN : v;
            colMax = Math.max(colMax, Math.abs(dv));
        }
        if (colMax == 0.0) {
            colMax = 1.0;
        }

        int est = n * 2 + (n / K_SEG + 2) * 64;
        ByteBuffer buf = ByteBuffer.allocate(est).order(ByteOrder.LITTLE_ENDIAN);
        double[] seg = new double[K_SEG];
        for (int off = 0; off < n; off += K_SEG) {
            int m = Math.min(K_SEG, n - off);
            for (int i = 0; i < m; i++) {
                Double v = col.get(off + i);
                seg[i] = v == null ? Double.NaN : v;
            }
            double[] ab = new double[2];
            olsAbDouble(seg, m, ab);
            double resMax = 0.0;
            double[] res = new double[m];
            for (int i = 0; i < m; i++) {
                double pred = ab[0] + ab[1] * i;
                res[i] = seg[i] - pred;
                resMax = Math.max(resMax, Math.abs(res[i]));
            }
            double scale = resMax > 0.0 ? resMax * REL_ERROR_DOUBLE : 1.0;
            double inv = 1.0 / scale;
            buf.putInt(m);
            buf.putDouble(ab[0]);
            buf.putDouble(ab[1]);
            buf.putDouble(scale);
            for (int i = 0; i < m; i++) {
                long qd = Math.round(res[i] * inv);
                if (qd > 32767) {
                    qd = 32767;
                }
                if (qd < -32768) {
                    qd = -32768;
                }
                buf.putShort((short) qd);
            }
        }
        return Arrays.copyOf(buf.array(), buf.position());
    }

    private static boolean verifyUnpackDouble(List<? extends Double> col, int n, byte[] raw) {
        double colMax = 0.0;
        for (int i = 0; i < n; i++) {
            Double v = col.get(i);
            double dv = v == null ? Double.NaN : v;
            colMax = Math.max(colMax, Math.abs(dv));
        }
        if (colMax == 0.0) {
            colMax = 1.0;
        }

        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (buf.remaining() >= 28) {
            int m = buf.getInt();
            double a = buf.getDouble();
            double b = buf.getDouble();
            double scale = buf.getDouble();
            if (m > K_SEG || off + m > n || buf.remaining() < m * 2) {
                return false;
            }
            for (int i = 0; i < m; i++) {
                short q = buf.getShort();
                double dec = a + b * i + q * scale;
                Double origObj = col.get(off + i);
                double orig = origObj == null ? Double.NaN : origObj;
                double err = Math.abs(dec - orig);
                double boundAbs = REL_ERROR_DOUBLE * Math.max(Math.abs(orig), colMax);
                if (err > boundAbs * 1.0001 + 1e-12) {
                    return false;
                }
            }
            off += m;
        }
        return off == n && !buf.hasRemaining();
    }

    public static void benchMostDoubleMinimal(List<? extends List<Double>> columns, long[] out) {
        benchMostDoubleMinimal(columns, out, null);
    }

    public static void benchMostDoubleMinimal(
            List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                byte[] raw = packColumnDouble(col, n);
                if (!verifyUnpackDouble(col, n, raw)) {
                    continue;
                }
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t2 = System.nanoTime();
                if (d.length != raw.length || !Arrays.equals(raw, d)) {
                    continue;
                }
                if (!verifyUnpackDouble(col, n, d)) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateUnpackDoubleLoss(col, n, d, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += c.length;
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

    private static void accumulateUnpackLongLoss(
            List<? extends Long> col, int n, byte[] raw, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return;
        }
        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (buf.remaining() >= 28) {
            int m = buf.getInt();
            double a = buf.getDouble();
            double b = buf.getDouble();
            double scale = buf.getDouble();
            if (m > K_SEG || off + m > n || buf.remaining() < m * 2) {
                return;
            }
            for (int i = 0; i < m; i++) {
                short q = buf.getShort();
                double dec = a + b * i + q * scale;
                Long lv = col.get(off + i);
                double orig = lv == null ? Double.NaN : lv.doubleValue();
                if (Double.isFinite(orig) && Double.isFinite(dec)) {
                    lossAcc[0] += Math.abs(orig - dec);
                    lossAcc[1] += Math.abs(orig);
                }
            }
            off += m;
        }
    }

    public static void benchMostLongMinimal(List<? extends List<Long>> columns, long[] out) {
        benchMostLongMinimal(columns, out, null);
    }

    public static void benchMostLongMinimal(
            List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                double colMax = 0.0;
                for (int i = 0; i < n; i++) {
                    Long lv = col.get(i);
                    long v = lv == null ? 0L : lv;
                    colMax = Math.max(colMax, Math.abs((double) v));
                }
                if (colMax == 0.0) {
                    colMax = 1.0;
                }
                int est = n * 2 + (n / K_SEG + 2) * 64;
                ByteBuffer buf = ByteBuffer.allocate(est).order(ByteOrder.LITTLE_ENDIAN);
                double[] seg = new double[K_SEG];
                for (int off = 0; off < n; off += K_SEG) {
                    int m = Math.min(K_SEG, n - off);
                    for (int i = 0; i < m; i++) {
                        Long lv = col.get(off + i);
                        seg[i] = lv == null ? 0L : lv;
                    }
                    double[] ab = new double[2];
                    olsAbDouble(seg, m, ab);
                    double resMax = 0.0;
                    double[] res = new double[m];
                    for (int i = 0; i < m; i++) {
                        double pred = ab[0] + ab[1] * i;
                        res[i] = seg[i] - pred;
                        resMax = Math.max(resMax, Math.abs(res[i]));
                    }
                    double scale = resMax > 0.0 ? resMax * REL_ERROR_DOUBLE : 1.0;
                    double inv = 1.0 / scale;
                    buf.putInt(m);
                    buf.putDouble(ab[0]);
                    buf.putDouble(ab[1]);
                    buf.putDouble(scale);
                    for (int i = 0; i < m; i++) {
                        long qd = Math.round(res[i] * inv);
                        if (qd > 32767) {
                            qd = 32767;
                        }
                        if (qd < -32768) {
                            qd = -32768;
                        }
                        buf.putShort((short) qd);
                    }
                }
                byte[] raw = Arrays.copyOf(buf.array(), buf.position());
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t2 = System.nanoTime();
                if (d.length != raw.length || !Arrays.equals(raw, d)) {
                    continue;
                }
                // Long path is lossy by design; verify packet self-consistency.
                ByteBuffer rb = ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN);
                int decoded = 0;
                boolean ok = true;
                while (rb.remaining() >= 28) {
                    int m = rb.getInt();
                    rb.getDouble();
                    rb.getDouble();
                    rb.getDouble();
                    if (m <= 0 || m > K_SEG || rb.remaining() < m * 2 || decoded + m > n) {
                        ok = false;
                        break;
                    }
                    rb.position(rb.position() + m * 2);
                    decoded += m;
                }
                if (!ok || decoded != n || rb.hasRemaining()) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateUnpackLongLoss(col, n, d, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += c.length;
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
