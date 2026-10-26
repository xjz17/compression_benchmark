package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

/**
 * CORAD-inspired tricklet baseline (matches {@code bench_corad_tricklet_float_columns} in Python).
 */
public final class CoradTrickletFloat {

    private static final int TRICK = 40;
    private static final int ATOMS = 4;

    private CoradTrickletFloat() {}

    private static byte[] zlibCompress(byte[] data, int level) {
        Deflater def = new Deflater(level);
        def.setInput(data);
        def.finish();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        while (!def.finished()) {
            int n = def.deflate(buf);
            if (n > 0) {
                bos.write(buf, 0, n);
            }
        }
        def.end();
        return bos.toByteArray();
    }

    private static byte[] zlibDecompress(byte[] compressed, int expectedLen) throws Exception {
        Inflater inf = new Inflater();
        inf.setInput(compressed);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        while (!inf.finished()) {
            int n = inf.inflate(buf);
            if (n > 0) {
                bos.write(buf, 0, n);
            }
        }
        inf.end();
        byte[] out = bos.toByteArray();
        if (out.length != expectedLen) {
            throw new IllegalStateException("zlib length mismatch");
        }
        return out;
    }

    private static void appendFloat32LE(ByteBuffer bb, float[][] rows) {
        for (float[] row : rows) {
            for (float v : row) {
                bb.putFloat(v);
            }
        }
    }

    private static void appendFloat64LE(ByteBuffer bb, double[][] rows) {
        for (double[] row : rows) {
            for (double v : row) {
                bb.putDouble(v);
            }
        }
    }

    private static byte[] constantFloat32Payload(int n, float value) {
        ByteBuffer bb =
                ByteBuffer.allocate(8 + Long.BYTES + Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
        bb.put("CORADC0F".getBytes(StandardCharsets.US_ASCII));
        bb.putLong(n);
        bb.putFloat(value);
        return bb.array();
    }

    private static byte[] constantFloat64Payload(int n, double value) {
        ByteBuffer bb =
                ByteBuffer.allocate(8 + Long.BYTES + Double.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
        bb.put("CORADC0D".getBytes(StandardCharsets.US_ASCII));
        bb.putLong(n);
        bb.putDouble(value);
        return bb.array();
    }

    private static byte[] constantInt64Payload(int n, long value) {
        ByteBuffer bb =
                ByteBuffer.allocate(8 + Long.BYTES + Long.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
        bb.put("CORADC0I".getBytes(StandardCharsets.US_ASCII));
        bb.putLong(n);
        bb.putLong(value);
        return bb.array();
    }

    public static void benchCoradTrickletFloat(List<? extends List<Float>> columns, long[] out) {
        benchCoradTrickletFloat(columns, out, null);
    }

    public static void benchCoradTrickletFloat(
            List<? extends List<Float>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Float> col : columns) {
                int n = col.size();
                if (n < TRICK * 2) {
                    continue;
                }
                double[] x = new double[n];
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    x[i] = col.get(i);
                    sum += x[i];
                }
                double mu = sum / n;
                double var = 0.0;
                for (int i = 0; i < n; i++) {
                    double d = x[i] - mu;
                    var += d * d;
                }
                double sig = Math.sqrt(var / n);
                byte[] payload;
                if (sig < 1e-15) {
                    payload = constantFloat32Payload(n, (float) x[0]);
                } else {
                    double[] z = new double[n];
                    for (int i = 0; i < n; i++) {
                        z[i] = (x[i] - mu) / sig;
                    }
                    int ntrim = (n / TRICK) * TRICK;
                    int nrows = ntrim / TRICK;
                    if (nrows < 1) {
                        continue;
                    }
                    float[][] tmat = new float[nrows][TRICK];
                    for (int r = 0; r < nrows; r++) {
                        for (int c = 0; c < TRICK; c++) {
                            tmat[r][c] = (float) z[r * TRICK + c];
                        }
                    }
                    int k = Math.min(ATOMS, nrows);
                    float[][] atomsMat = new float[k][TRICK];
                    for (int r = 0; r < k; r++) {
                        System.arraycopy(tmat[r], 0, atomsMat[r], 0, TRICK);
                    }
                    int payloadLen = k * TRICK * 4 + nrows * TRICK * 4;
                    ByteBuffer bb = ByteBuffer.allocate(payloadLen).order(ByteOrder.LITTLE_ENDIAN);
                    appendFloat32LE(bb, atomsMat);
                    appendFloat32LE(bb, tmat);
                    payload = bb.array();
                }

                long t0 = System.nanoTime();
                byte[] compressed = zlibCompress(payload, 6);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] dec = zlibDecompress(compressed, payload.length);
                long t3 = System.nanoTime();
                if (!java.util.Arrays.equals(dec, payload)) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compressed.length;
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

    public static void benchCoradTrickletDouble(List<? extends List<Double>> columns, long[] out) {
        benchCoradTrickletDouble(columns, out, null);
    }

    public static void benchCoradTrickletDouble(
            List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n < TRICK * 2) {
                    continue;
                }
                double[] x = new double[n];
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    Double dv = col.get(i);
                    x[i] = dv == null ? Double.NaN : dv;
                    sum += x[i];
                }
                double mu = sum / n;
                double var = 0.0;
                for (int i = 0; i < n; i++) {
                    double d = x[i] - mu;
                    var += d * d;
                }
                double sig = Math.sqrt(var / n);
                byte[] payload;
                if (sig < 1e-15) {
                    payload = constantFloat64Payload(n, x[0]);
                } else {
                    double[] z = new double[n];
                    for (int i = 0; i < n; i++) {
                        z[i] = (x[i] - mu) / sig;
                    }
                    int ntrim = (n / TRICK) * TRICK;
                    int nrows = ntrim / TRICK;
                    if (nrows < 1) {
                        continue;
                    }
                    double[][] tmat = new double[nrows][TRICK];
                    for (int r = 0; r < nrows; r++) {
                        for (int c = 0; c < TRICK; c++) {
                            tmat[r][c] = z[r * TRICK + c];
                        }
                    }
                    int k = Math.min(ATOMS, nrows);
                    double[][] atomsMat = new double[k][TRICK];
                    for (int r = 0; r < k; r++) {
                        System.arraycopy(tmat[r], 0, atomsMat[r], 0, TRICK);
                    }
                    int payloadLen = k * TRICK * 8 + nrows * TRICK * 8;
                    ByteBuffer bb = ByteBuffer.allocate(payloadLen).order(ByteOrder.LITTLE_ENDIAN);
                    appendFloat64LE(bb, atomsMat);
                    appendFloat64LE(bb, tmat);
                    payload = bb.array();
                }

                long t0 = System.nanoTime();
                byte[] compressed = zlibCompress(payload, 6);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] dec = zlibDecompress(compressed, payload.length);
                long t3 = System.nanoTime();
                if (!java.util.Arrays.equals(dec, payload)) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compressed.length;
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

    public static void benchCoradTrickletLong(List<? extends List<Long>> columns, long[] out) {
        benchCoradTrickletLong(columns, out, null);
    }

    public static void benchCoradTrickletLong(
            List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0, comp = 0, ct = 0, dt = 0;
        try {
            for (List<? extends Long> col : columns) {
                int n = col.size();
                if (n < TRICK * 2) {
                    continue;
                }
                double[] x = new double[n];
                double sum = 0.0;
                for (int i = 0; i < n; i++) {
                    Long lv = col.get(i);
                    x[i] = lv == null ? 0L : lv;
                    sum += x[i];
                }
                double mu = sum / n;
                double var = 0.0;
                for (int i = 0; i < n; i++) {
                    double d = x[i] - mu;
                    var += d * d;
                }
                double sig = Math.sqrt(var / n);
                byte[] payload;
                if (sig < 1e-15) {
                    payload = constantInt64Payload(n, col.get(0) == null ? 0L : col.get(0));
                } else {
                    double[] z = new double[n];
                    for (int i = 0; i < n; i++) {
                        z[i] = (x[i] - mu) / sig;
                    }
                    int ntrim = (n / TRICK) * TRICK;
                    int nrows = ntrim / TRICK;
                    if (nrows < 1) {
                        continue;
                    }
                    double[][] tmat = new double[nrows][TRICK];
                    for (int r = 0; r < nrows; r++) {
                        for (int c = 0; c < TRICK; c++) {
                            tmat[r][c] = z[r * TRICK + c];
                        }
                    }
                    int k = Math.min(ATOMS, nrows);
                    double[][] atomsMat = new double[k][TRICK];
                    for (int r = 0; r < k; r++) {
                        System.arraycopy(tmat[r], 0, atomsMat[r], 0, TRICK);
                    }
                    int payloadLen = k * TRICK * 8 + nrows * TRICK * 8;
                    ByteBuffer bb = ByteBuffer.allocate(payloadLen).order(ByteOrder.LITTLE_ENDIAN);
                    appendFloat64LE(bb, atomsMat);
                    appendFloat64LE(bb, tmat);
                    payload = bb.array();
                }

                long t0 = System.nanoTime();
                byte[] compressed = zlibCompress(payload, 6);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                byte[] dec = zlibDecompress(compressed, payload.length);
                long t3 = System.nanoTime();
                if (!java.util.Arrays.equals(dec, payload)) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compressed.length;
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
