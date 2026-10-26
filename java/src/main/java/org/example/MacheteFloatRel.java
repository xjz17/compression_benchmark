package org.example;

import com.github.luben.zstd.Zstd;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Uniform int16 quantizer + ZSTD (lossy); CSV keys {@code machete_float_rel_1e-1} … {@code _1e-4}.
 */
public final class MacheteFloatRel {

    private MacheteFloatRel() {}

    public static void benchMacheteFloatRel(
            List<? extends List<Float>> columns, double relError, long[] out) {
        benchMacheteFloatRel(columns, relError, out, null);
    }

    public static void benchMacheteFloatRel(
            List<? extends List<Float>> columns, double relError, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (!(relError > 0.0) || relError >= 1.0) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                float mx = 0f;
                for (float v : col) {
                    mx = Math.max(mx, Math.abs(v));
                }
                if (mx == 0f) {
                    mx = 1f;
                }
                float scale = (float) ((double) mx * relError);
                if (!(scale > 0f)) {
                    continue;
                }
                float inv = 1f / scale;
                short[] q = new short[n];
                for (int i = 0; i < n; i++) {
                    long d = Math.round((double) col.get(i) * (double) inv);
                    if (d > 32767) {
                        d = 32767;
                    }
                    if (d < -32768) {
                        d = -32768;
                    }
                    q[i] = (short) d;
                }
                byte[] raw = new byte[8 + n * 2];
                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, scale).putInt(4, n);
                for (int i = 0; i < n; i++) {
                    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).putShort(8 + i * 2, q[i]);
                }
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t3 = System.nanoTime();
                if (d.length != raw.length) {
                    continue;
                }
                float s2 = ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN).getFloat(0);
                int n2 = ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN).getInt(4);
                if (n2 != n || Float.floatToIntBits(s2) != Float.floatToIntBits(scale)) {
                    continue;
                }
                boolean okCol = true;
                for (int i = 0; i < n; i++) {
                    short q2 =
                            ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN).getShort(8 + i * 2);
                    float dec = q2 * s2;
                    float err = Math.abs(dec - col.get(i));
                    float boundAbs = (float) (relError * Math.max(Math.abs(col.get(i)), mx));
                    if (err > boundAbs * 1.0001f + 1e-6f) {
                        okCol = false;
                        break;
                    }
                    if (lossAcc != null) {
                        lossAcc[0] += err;
                        lossAcc[1] += Math.abs(col.get(i));
                    }
                }
                if (!okCol) {
                    continue;
                }
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

    public static void benchMacheteDoubleRel(
            List<? extends List<Double>> columns, double relError, long[] out) {
        benchMacheteDoubleRel(columns, relError, out, null);
    }

    public static void benchMacheteDoubleRel(
            List<? extends List<Double>> columns, double relError, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (!(relError > 0.0) || relError >= 1.0) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                double mx = 0.0;
                for (Double dv : col) {
                    double v = dv == null ? Double.NaN : dv;
                    mx = Math.max(mx, Math.abs(v));
                }
                if (mx == 0.0) {
                    mx = 1.0;
                }
                double scale = mx * relError;
                if (!(scale > 0.0)) {
                    continue;
                }
                double inv = 1.0 / scale;
                short[] q = new short[n];
                for (int i = 0; i < n; i++) {
                    Double dv = col.get(i);
                    double v = dv == null ? Double.NaN : dv;
                    long d = Math.round(v * inv);
                    if (d > 32767) {
                        d = 32767;
                    }
                    if (d < -32768) {
                        d = -32768;
                    }
                    q[i] = (short) d;
                }
                byte[] raw = new byte[12 + n * 2];
                ByteBuffer hb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                hb.putDouble(0, scale).putInt(8, n);
                for (int i = 0; i < n; i++) {
                    hb.putShort(12 + i * 2, q[i]);
                }
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t3 = System.nanoTime();
                if (d.length != raw.length) {
                    continue;
                }
                ByteBuffer db = ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN);
                double s2 = db.getDouble(0);
                int n2 = db.getInt(8);
                if (n2 != n || Double.doubleToRawLongBits(s2) != Double.doubleToRawLongBits(scale)) {
                    continue;
                }
                boolean okCol = true;
                for (int i = 0; i < n; i++) {
                    short q2 = db.getShort(12 + i * 2);
                    double dec = q2 * s2;
                    Double ov = col.get(i);
                    double origV = ov == null ? Double.NaN : ov;
                    double err = Math.abs(dec - origV);
                    double boundAbs = relError * Math.max(Math.abs(origV), mx);
                    if (err > boundAbs * 1.0001 + 1e-12) {
                        okCol = false;
                        break;
                    }
                    if (lossAcc != null && Double.isFinite(origV)) {
                        lossAcc[0] += err;
                        lossAcc[1] += Math.abs(origV);
                    }
                }
                if (!okCol) {
                    continue;
                }
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

    public static void benchMacheteLongRel(
            List<? extends List<Long>> columns, double relError, long[] out) {
        benchMacheteLongRel(columns, relError, out, null);
    }

    public static void benchMacheteLongRel(
            List<? extends List<Long>> columns, double relError, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        if (!(relError > 0.0) || relError >= 1.0) {
            out[0] = 0;
            return;
        }
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                double mx = 0.0;
                for (Long lv : col) {
                    long v = lv == null ? 0L : lv;
                    mx = Math.max(mx, Math.abs((double) v));
                }
                if (mx == 0.0) {
                    mx = 1.0;
                }
                double scale = mx * relError;
                if (!(scale > 0.0)) {
                    continue;
                }
                double inv = 1.0 / scale;
                short[] q = new short[n];
                for (int i = 0; i < n; i++) {
                    Long lv = col.get(i);
                    long v = lv == null ? 0L : lv;
                    long d = Math.round(v * inv);
                    if (d > 32767) {
                        d = 32767;
                    }
                    if (d < -32768) {
                        d = -32768;
                    }
                    q[i] = (short) d;
                }
                byte[] raw = new byte[12 + n * 2];
                ByteBuffer hb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                hb.putDouble(0, scale).putInt(8, n);
                for (int i = 0; i < n; i++) {
                    hb.putShort(12 + i * 2, q[i]);
                }
                long t0 = System.nanoTime();
                byte[] c = Zstd.compress(raw, 3);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] d = Zstd.decompress(c, raw.length);
                long t3 = System.nanoTime();
                if (d.length != raw.length) {
                    continue;
                }
                ByteBuffer db = ByteBuffer.wrap(d).order(ByteOrder.LITTLE_ENDIAN);
                double s2 = db.getDouble(0);
                int n2 = db.getInt(8);
                if (n2 != n || Double.doubleToRawLongBits(s2) != Double.doubleToRawLongBits(scale)) {
                    continue;
                }
                boolean okCol = true;
                for (int i = 0; i < n; i++) {
                    short q2 = db.getShort(12 + i * 2);
                    double dec = q2 * s2;
                    Long lv = col.get(i);
                    double origV = lv == null ? 0.0 : lv.doubleValue();
                    double err = Math.abs(dec - origV);
                    double boundAbs = relError * Math.max(Math.abs(origV), mx);
                    if (err > boundAbs * 1.0001 + 1.0) {
                        okCol = false;
                        break;
                    }
                    if (lossAcc != null) {
                        lossAcc[0] += err;
                        lossAcc[1] += Math.abs(origV);
                    }
                }
                if (!okCol) {
                    continue;
                }
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
