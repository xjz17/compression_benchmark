package org.example;

import com.github.luben.zstd.Zstd;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Sim-Piece float codec aligned with {@code backend/myapp/simpiece_python.py} and
 * {@code cpp/src/simpiece_float_codec.cpp} (segment logic + Zstd level 3).
 */
public final class SimPieceFloatNative {

    private static final double DBL_MAX = Double.MAX_VALUE;

    private SimPieceFloatNative() {}

    private static final class Seg {
        final long initTs;
        final double aMin;
        final double aMax;
        final double b;

        Seg(long initTs, double aMin, double aMax, double b) {
            this.initTs = initTs;
            this.aMin = aMin;
            this.aMax = aMax;
            this.b = b;
        }

        double aMid() {
            return (aMin + aMax) / 2.0;
        }
    }

    private static double quantize(double value, double epsilon) {
        return Math.round(value / epsilon) * epsilon;
    }

    private static void vbWriteU32(int n, ByteBuffer out) {
        long val = n & 0xFFFFFFFFL;
        if (val < (1 << 7)) {
            out.put((byte) ((val | (1 << 7)) & 0xFF));
        } else if (val < (1 << 14)) {
            out.put((byte) (ex7(val, 0) & 0xFF));
            out.put((byte) ((exm(val, 1) | (1 << 7)) & 0xFF));
        } else if (val < (1 << 21)) {
            out.put((byte) (ex7(val, 0) & 0xFF));
            out.put((byte) (ex7(val, 1) & 0xFF));
            out.put((byte) ((exm(val, 2) | (1 << 7)) & 0xFF));
        } else if (val < (1 << 28)) {
            out.put((byte) (ex7(val, 0) & 0xFF));
            out.put((byte) (ex7(val, 1) & 0xFF));
            out.put((byte) (ex7(val, 2) & 0xFF));
            out.put((byte) ((exm(val, 3) | (1 << 7)) & 0xFF));
        } else {
            out.put((byte) (ex7(val, 0) & 0xFF));
            out.put((byte) (ex7(val, 1) & 0xFF));
            out.put((byte) (ex7(val, 2) & 0xFF));
            out.put((byte) (ex7(val, 3) & 0xFF));
            out.put((byte) ((exm(val, 4) | (1 << 7)) & 0xFF));
        }
    }

    private static int ex7(long val, int i) {
        return (int) ((val >> (7 * i)) & ((1 << 7) - 1));
    }

    private static int exm(long val, int i) {
        return (int) ((val >> (7 * i)) & 0xFF);
    }

    private static int vbReadU32(ByteBuffer in) {
        int inn = in.get() & 0xFF;
        int number = inn & 0x7F;
        if (inn >= 0x80) {
            return number;
        }
        inn = in.get() & 0xFF;
        number = ((inn & 0x7F) << 7) | number;
        if (inn >= 0x80) {
            return number;
        }
        inn = in.get() & 0xFF;
        number = ((inn & 0x7F) << 14) | number;
        if (inn >= 0x80) {
            return number;
        }
        inn = in.get() & 0xFF;
        number = ((inn & 0x7F) << 21) | number;
        if (inn >= 0x80) {
            return number;
        }
        inn = in.get() & 0xFF;
        return (((inn & 0x7F) << 28) | number) & 0xFFFFFFFF;
    }

    private static void writeF32Be(float x, ByteBuffer out) {
        out.order(ByteOrder.BIG_ENDIAN).putFloat(x);
    }

    private static float readF32Be(ByteBuffer in) {
        return in.order(ByteOrder.BIG_ENDIAN).getFloat();
    }

    private static int createSegment(
            int startIdx, long[] ts, double[] vals, double epsilon, List<Seg> segs) {
        int n = vals.length;
        long initTs = ts[startIdx];
        double b = quantize(vals[startIdx], epsilon);
        if (startIdx + 1 == n) {
            segs.add(new Seg(initTs, -DBL_MAX, DBL_MAX, b));
            return startIdx + 1;
        }
        double dt = ts[startIdx + 1] - initTs;
        double aMax = ((vals[startIdx + 1] + epsilon) - b) / dt;
        double aMin = ((vals[startIdx + 1] - epsilon) - b) / dt;
        if (startIdx + 2 == n) {
            segs.add(new Seg(initTs, aMin, aMax, b));
            return startIdx + 2;
        }
        for (int idx = startIdx + 2; idx < n; idx++) {
            double upVal = vals[idx] + epsilon;
            double downVal = vals[idx] - epsilon;
            double dtI = ts[idx] - initTs;
            double upLim = aMax * dtI + b;
            double downLim = aMin * dtI + b;
            if (downVal > upLim || upVal < downLim) {
                segs.add(new Seg(initTs, aMin, aMax, b));
                return idx;
            }
            if (upVal < upLim) {
                aMax = Math.max((upVal - b) / dtI, aMin);
            }
            if (downVal > downLim) {
                aMin = Math.min((downVal - b) / dtI, aMax);
            }
        }
        segs.add(new Seg(initTs, aMin, aMax, b));
        return n;
    }

    private static List<Seg> compressSegments(long[] ts, double[] vals, double epsilon) {
        List<Seg> segs = new ArrayList<>();
        int cur = 0;
        int n = vals.length;
        while (cur < n) {
            cur = createSegment(cur, ts, vals, epsilon, segs);
        }
        return segs;
    }

    private static List<Seg> mergePerB(List<Seg> segments) {
        double aMinTemp = -DBL_MAX;
        double aMaxTemp = DBL_MAX;
        Double bCur = null;
        List<Long> timestamps = new ArrayList<>();
        List<Seg> merged = new ArrayList<>();
        List<Seg> segs = new ArrayList<>(segments);
        segs.sort(Comparator.comparingDouble((Seg s) -> s.b).thenComparingDouble(s -> s.aMid()));
        for (int i = 0; i < segs.size(); i++) {
            Seg sg = segs.get(i);
            if (bCur == null || Double.compare(bCur, sg.b) != 0) {
                if (bCur != null) {
                    if (timestamps.size() == 1) {
                        merged.add(new Seg(timestamps.get(0), aMinTemp, aMaxTemp, bCur));
                    } else if (timestamps.size() > 1) {
                        for (long t : timestamps) {
                            merged.add(new Seg(t, aMinTemp, aMaxTemp, bCur));
                        }
                    }
                }
                timestamps.clear();
                timestamps.add(sg.initTs);
                aMinTemp = sg.aMin;
                aMaxTemp = sg.aMax;
                bCur = sg.b;
                continue;
            }
            if (sg.aMin <= aMaxTemp && sg.aMax >= aMinTemp) {
                timestamps.add(sg.initTs);
                aMinTemp = Math.max(aMinTemp, sg.aMin);
                aMaxTemp = Math.min(aMaxTemp, sg.aMax);
            } else {
                if (timestamps.size() == 1) {
                    merged.add(segs.get(i - 1));
                } else {
                    for (long t : timestamps) {
                        merged.add(new Seg(t, aMinTemp, aMaxTemp, bCur));
                    }
                }
                timestamps.clear();
                timestamps.add(sg.initTs);
                aMinTemp = sg.aMin;
                aMaxTemp = sg.aMax;
            }
        }
        if (bCur != null && !timestamps.isEmpty()) {
            if (timestamps.size() == 1) {
                merged.add(new Seg(timestamps.get(0), aMinTemp, aMaxTemp, bCur));
            } else {
                for (long t : timestamps) {
                    merged.add(new Seg(t, aMinTemp, aMaxTemp, bCur));
                }
            }
        }
        return merged;
    }

    private static final class TsVal {
        final int ts;
        final double val;

        TsVal(int ts, double val) {
            this.ts = ts;
            this.val = val;
        }
    }

    private static List<TsVal> toPoints(List<Seg> segments, int lastTs) {
        List<Seg> segs = new ArrayList<>(segments);
        segs.sort(Comparator.comparingLong(s -> s.initTs));
        if (segs.isEmpty()) {
            return List.of();
        }
        List<TsVal> out = new ArrayList<>();
        long cur = segs.get(0).initTs;
        for (int i = 0; i + 1 < segs.size(); i++) {
            Seg sg = segs.get(i);
            long nxt = segs.get(i + 1).initTs;
            while (cur < nxt) {
                double v = sg.aMid() * (cur - sg.initTs) + sg.b;
                out.add(new TsVal((int) cur, v));
                cur++;
            }
        }
        Seg last = segs.get(segs.size() - 1);
        while (cur <= lastTs) {
            double v = last.aMid() * (cur - last.initTs) + last.b;
            out.add(new TsVal((int) cur, v));
            cur++;
        }
        return out;
    }

    private static byte[] toByteArrayPerB(List<Seg> segments, double epsilon) {
        TreeMap<Integer, TreeMap<Double, List<Long>>> tree = new TreeMap<>();
        for (Seg sg : segments) {
            double a = sg.aMid();
            int bInt = (int) Math.round(sg.b / epsilon);
            tree.computeIfAbsent(bInt, k -> new TreeMap<>()).computeIfAbsent(a, k -> new ArrayList<>()).add(sg.initTs);
        }
        ByteBuffer buf = ByteBuffer.allocate(1 << 20).order(ByteOrder.BIG_ENDIAN);
        vbWriteU32(tree.size(), buf);
        if (tree.isEmpty()) {
            byte[] raw = new byte[buf.position()];
            buf.flip();
            buf.get(raw);
            return raw;
        }
        int prevB = tree.firstKey();
        vbWriteU32(prevB, buf);
        for (Map.Entry<Integer, TreeMap<Double, List<Long>>> be : tree.entrySet()) {
            vbWriteU32(be.getKey() - prevB, buf);
            prevB = be.getKey();
            TreeMap<Double, List<Long>> aMap = be.getValue();
            vbWriteU32(aMap.size(), buf);
            for (Map.Entry<Double, List<Long>> ae : aMap.entrySet()) {
                writeF32Be(ae.getKey().floatValue(), buf);
                List<Long> tsList = new ArrayList<>(ae.getValue());
                tsList.sort(Long::compare);
                vbWriteU32(tsList.size(), buf);
                long prevTs = 0;
                for (long tstamp : tsList) {
                    vbWriteU32((int) (tstamp - prevTs), buf);
                    prevTs = tstamp;
                }
            }
        }
        byte[] raw = new byte[buf.position()];
        buf.flip();
        buf.get(raw);
        return raw;
    }

    private static List<Seg> readMergedPerB(ByteBuffer stream, double epsilon) {
        List<Seg> segments = new ArrayList<>();
        int numB = vbReadU32(stream);
        if (numB == 0) {
            return segments;
        }
        int prevB = vbReadU32(stream);
        for (int i = 0; i < numB; i++) {
            int b = vbReadU32(stream) + prevB;
            prevB = b;
            int numA = vbReadU32(stream);
            for (int j = 0; j < numA; j++) {
                float a = readF32Be(stream);
                int nTs = vbReadU32(stream);
                long timestamp = 0;
                for (int k = 0; k < nTs; k++) {
                    timestamp += Integer.toUnsignedLong(vbReadU32(stream));
                    segments.add(new Seg(timestamp, a, a, b * epsilon));
                }
            }
        }
        return segments;
    }

    public static byte[] compressColumn(List<Float> col, double relRangeFrac) {
        int n = col.size();
        if (n == 0) {
            throw new IllegalArgumentException("empty column");
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (float f : col) {
            double v = f;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double range = max - min;
        if (range <= 0 || Double.isNaN(range)) {
            range = 1.0;
        }
        double epsilon = range * relRangeFrac;
        if (epsilon <= 0) {
            throw new IllegalArgumentException("epsilon");
        }
        return compressInner(col, epsilon);
    }

    private static byte[] compressInner(List<Float> col, double epsilon) {
        int n = col.size();
        long[] ts = new long[n];
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            ts[i] = i;
            vals[i] = col.get(i).doubleValue();
        }
        List<Seg> rawSegs = compressSegments(ts, vals, epsilon);
        List<Seg> merged = mergePerB(rawSegs);
        ByteBuffer preamble = ByteBuffer.allocate(1 << 20).order(ByteOrder.BIG_ENDIAN);
        writeF32Be((float) epsilon, preamble);
        byte[] perB = toByteArrayPerB(merged, epsilon);
        preamble.put(perB);
        vbWriteU32(n - 1, preamble);
        byte[] pre = new byte[preamble.position()];
        preamble.flip();
        preamble.get(pre);
        return Zstd.compress(pre, 3);
    }

    private static List<TsVal> decompressColumn(byte[] packed) {
        long frame = Zstd.decompressedSize(packed);
        int cap = frame > 0 && frame < Integer.MAX_VALUE ? (int) frame : Math.max(packed.length * 16, 65536);
        byte[] raw = Zstd.decompress(packed, cap);
        ByteBuffer stream = ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN);
        float epsF = readF32Be(stream);
        double epsilon = epsF;
        List<Seg> segs = readMergedPerB(stream, epsilon);
        int lastTs = vbReadU32(stream);
        return toPoints(segs, lastTs);
    }

    private static byte[] compressColumnDoubleValues(double[] values, double relRangeFrac) {
        int n = values.length;
        if (n == 0) {
            throw new IllegalArgumentException("empty column");
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : values) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double range = max - min;
        if (range <= 0 || Double.isNaN(range)) {
            range = 1.0;
        }
        double epsilon = range * relRangeFrac;
        if (epsilon <= 0) {
            throw new IllegalArgumentException("epsilon");
        }
        long[] ts = new long[n];
        for (int i = 0; i < n; i++) {
            ts[i] = i;
        }
        List<Seg> rawSegs = compressSegments(ts, values, epsilon);
        List<Seg> merged = mergePerB(rawSegs);
        ByteBuffer preamble = ByteBuffer.allocate(1 << 20).order(ByteOrder.BIG_ENDIAN);
        writeF32Be((float) epsilon, preamble);
        byte[] perB = toByteArrayPerB(merged, epsilon);
        preamble.put(perB);
        vbWriteU32(n - 1, preamble);
        byte[] pre = new byte[preamble.position()];
        preamble.flip();
        preamble.get(pre);
        return Zstd.compress(pre, 3);
    }

    private static boolean verifyToleranceDoubleValues(
            byte[] packed, double[] values, double relRangeFrac) {
        int n = values.length;
        if (n == 0) {
            return false;
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double v : values) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double range = max - min;
        if (range <= 0 || Double.isNaN(range)) {
            range = 1.0;
        }
        double epsilon = range * relRangeFrac;
        if (epsilon <= 0) {
            return false;
        }
        try {
            List<TsVal> restored = decompressColumn(packed);
            double[] byTs = new double[n];
            for (int i = 0; i < n; i++) {
                byTs[i] = Double.NaN;
            }
            for (TsVal pr : restored) {
                if (pr.ts >= 0 && pr.ts < n) {
                    byTs[pr.ts] = pr.val;
                }
            }
            double tol = epsilon * 1.0001 + 1e-12;
            for (int i = 0; i < n; i++) {
                double dv = byTs[i];
                if (Double.isNaN(dv) || Math.abs(dv - values[i]) > tol) {
                    return false;
                }
            }
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    static boolean verifyTolerance(byte[] packed, List<Float> col, double relRangeFrac) {
        int n = col.size();
        if (n == 0) {
            return false;
        }
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (float f : col) {
            double v = f;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double range = max - min;
        if (range <= 0 || Double.isNaN(range)) {
            range = 1.0;
        }
        double epsilon = range * relRangeFrac;
        if (epsilon <= 0) {
            return false;
        }
        try {
            List<TsVal> restored = decompressColumn(packed);
            double[] byTs = new double[n];
            for (int i = 0; i < n; i++) {
                byTs[i] = Double.NaN;
            }
            for (TsVal pr : restored) {
                if (pr.ts >= 0 && pr.ts < n) {
                    byTs[pr.ts] = pr.val;
                }
            }
            double tol = epsilon * 1.0001 + 1e-9;
            for (int i = 0; i < n; i++) {
                double dv = byTs[i];
                if (Double.isNaN(dv) || Math.abs(dv - col.get(i).doubleValue()) > tol) {
                    return false;
                }
            }
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    static boolean benchFloatColumnsRelRange(
            List<ArrayList<Float>> columns, double relRangeFrac, long[] out) {
        return benchFloatColumnsRelRange(columns, relRangeFrac, out, null);
    }

    static boolean benchFloatColumnsRelRange(
            List<ArrayList<Float>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        for (List<Float> col : columns) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            int n = col.size();
            try {
                long t0 = System.nanoTime();
                byte[] packed = compressColumn(col, relRangeFrac);
                long t1 = System.nanoTime();
                if (!verifyTolerance(packed, col, relRangeFrac)) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateDecompressedDoubleLoss(packed, col, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += packed.length;
                ct += t1 - t0;
                dt += t3 - t1;
            } catch (RuntimeException ignored) {
                // skip column
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    static boolean benchDoubleColumnsRelRange(
            List<? extends List<Double>> columns, double relRangeFrac, long[] out) {
        return benchDoubleColumnsRelRange(columns, relRangeFrac, out, null);
    }

    static boolean benchDoubleColumnsRelRange(
            List<? extends List<Double>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        for (List<Double> col : columns) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            int n = col.size();
            try {
                double[] values = new double[n];
                for (int i = 0; i < n; i++) {
                    Double dv = col.get(i);
                    values[i] = dv == null ? Double.NaN : dv;
                }
                long t0 = System.nanoTime();
                byte[] packed = compressColumnDoubleValues(values, relRangeFrac);
                long t1 = System.nanoTime();
                if (!verifyToleranceDoubleValues(packed, values, relRangeFrac)) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateDecompressedDoubleValuesLoss(packed, values, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += packed.length;
                ct += t1 - t0;
                dt += t3 - t1;
            } catch (RuntimeException ignored) {
                // skip column
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    static boolean benchLongColumnsRelRange(
            List<? extends List<Long>> columns, double relRangeFrac, long[] out) {
        return benchLongColumnsRelRange(columns, relRangeFrac, out, null);
    }

    static boolean benchLongColumnsRelRange(
            List<? extends List<Long>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        for (List<Long> col : columns) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            int n = col.size();
            try {
                double[] values = new double[n];
                for (int i = 0; i < n; i++) {
                    Long lv = col.get(i);
                    values[i] = lv == null ? 0L : lv.doubleValue();
                }
                long t0 = System.nanoTime();
                byte[] packed = compressColumnDoubleValues(values, relRangeFrac);
                long t1 = System.nanoTime();
                if (!verifyToleranceDoubleValues(packed, values, relRangeFrac)) {
                    continue;
                }
                if (lossAcc != null) {
                    accumulateDecompressedDoubleValuesLoss(packed, values, lossAcc);
                }
                long t3 = System.nanoTime();
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += packed.length;
                ct += t1 - t0;
                dt += t3 - t1;
            } catch (RuntimeException ignored) {
                // skip column
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    private static void accumulateDecompressedDoubleLoss(
            byte[] packed, List<Float> col, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return;
        }
        List<TsVal> restored = decompressColumn(packed);
        double[] byTs = new double[col.size()];
        java.util.Arrays.fill(byTs, Double.NaN);
        for (TsVal pr : restored) {
            if (pr.ts >= 0 && pr.ts < byTs.length) {
                byTs[pr.ts] = pr.val;
            }
        }
        for (int i = 0; i < col.size(); i++) {
            double o = col.get(i).doubleValue();
            double d = byTs[i];
            if (Double.isFinite(o) && Double.isFinite(d)) {
                lossAcc[0] += Math.abs(o - d);
                lossAcc[1] += Math.abs(o);
            }
        }
    }

    private static void accumulateDecompressedDoubleValuesLoss(
            byte[] packed, double[] values, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return;
        }
        List<TsVal> restored = decompressColumn(packed);
        double[] byTs = new double[values.length];
        java.util.Arrays.fill(byTs, Double.NaN);
        for (TsVal pr : restored) {
            if (pr.ts >= 0 && pr.ts < byTs.length) {
                byTs[pr.ts] = pr.val;
            }
        }
        for (int i = 0; i < values.length; i++) {
            double o = values[i];
            double d = byTs[i];
            if (Double.isFinite(o) && Double.isFinite(d)) {
                lossAcc[0] += Math.abs(o - d);
                lossAcc[1] += Math.abs(o);
            }
        }
    }
}
