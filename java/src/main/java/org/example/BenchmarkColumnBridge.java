package org.example;

import java.util.ArrayList;

/** Narrow/widen helpers for legacy float32 research benches behind int64/double wire format. */
public final class BenchmarkColumnBridge {

    private BenchmarkColumnBridge() {}

    public static ArrayList<ArrayList<Integer>> toIntegerColumns(
            ArrayList<ArrayList<Long>> in) {
        ArrayList<ArrayList<Integer>> out = new ArrayList<>();
        if (in == null) {
            return out;
        }
        for (ArrayList<Long> col : in) {
            ArrayList<Integer> ic = new ArrayList<>(col.size());
            for (Long v : col) {
                ic.add(v == null ? 0 : v.intValue());
            }
            out.add(ic);
        }
        return out;
    }

    public static ArrayList<ArrayList<Long>> toLongColumns(
            ArrayList<ArrayList<Integer>> in) {
        ArrayList<ArrayList<Long>> out = new ArrayList<>();
        if (in == null) {
            return out;
        }
        for (ArrayList<Integer> col : in) {
            ArrayList<Long> lc = new ArrayList<>(col.size());
            for (Integer v : col) {
                lc.add(v == null ? 0L : v.longValue());
            }
            out.add(lc);
        }
        return out;
    }

    public static ArrayList<ArrayList<Double>> toDoubleColumns(
            ArrayList<ArrayList<Float>> in) {
        ArrayList<ArrayList<Double>> out = new ArrayList<>();
        if (in == null) {
            return out;
        }
        for (ArrayList<Float> col : in) {
            ArrayList<Double> dc = new ArrayList<>(col.size());
            for (Float v : col) {
                dc.add(v == null ? Double.NaN : v.doubleValue());
            }
            out.add(dc);
        }
        return out;
    }

    public static ArrayList<ArrayList<Float>> toFloatColumns(
            ArrayList<ArrayList<Double>> in) {
        ArrayList<ArrayList<Float>> out = new ArrayList<>();
        if (in == null) {
            return out;
        }
        for (ArrayList<Double> col : in) {
            ArrayList<Float> fc = new ArrayList<>(col.size());
            for (Double v : col) {
                fc.add(v == null ? Float.NaN : v.floatValue());
            }
            out.add(fc);
        }
        return out;
    }
}
