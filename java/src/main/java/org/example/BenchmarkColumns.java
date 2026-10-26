package org.example;

import org.apache.tsfile.enums.TSDataType;

import java.util.ArrayList;

/** Column list helpers for cross-type benchmark rows (int columns as double values). */
public final class BenchmarkColumns {

    private BenchmarkColumns() {}

    /**
     * Treat each int64 as a real-valued double with zero fractional part ({@code (double) longValue}),
     * not {@code Double.longBitsToDouble}.
     */
    public static ArrayList<ArrayList<Double>> integersAsDoubleColumns(
            ArrayList<ArrayList<Long>> intCols) {
        ArrayList<ArrayList<Double>> out = new ArrayList<>();
        if (intCols == null) {
            return out;
        }
        for (ArrayList<Long> col : intCols) {
            ArrayList<Double> fcol = new ArrayList<>(col.size());
            for (Long v : col) {
                fcol.add((double) v.longValue());
            }
            out.add(fcol);
        }
        return out;
    }

    /** @deprecated Use {@link #integersAsDoubleColumns}. */
    public static ArrayList<ArrayList<Float>> integersAsFloatColumns(
            ArrayList<ArrayList<Integer>> intCols) {
        ArrayList<ArrayList<Float>> out = new ArrayList<>();
        if (intCols == null) {
            return out;
        }
        for (ArrayList<Integer> col : intCols) {
            ArrayList<Float> fcol = new ArrayList<>(col.size());
            for (Integer v : col) {
                fcol.add((float) v.intValue());
            }
            out.add(fcol);
        }
        return out;
    }

    /** Double columns for float-targeted research codecs on the current benchmark run. */
    public static ArrayList<ArrayList<Double>> doubleColumnsForCodecBench(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols) {
        if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
            return floatCols;
        }
        if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
            return integersAsDoubleColumns(intCols);
        }
        return new ArrayList<>();
    }

    /** Cast int64 columns to float32 views for legacy float-targeted research CLIs (fpzip, SZ3, BUFF, …). */
    public static ArrayList<ArrayList<Float>> longColumnsAsFloatColumns(
            ArrayList<ArrayList<Long>> intCols) {
        ArrayList<ArrayList<Float>> out = new ArrayList<>();
        if (intCols == null) {
            return out;
        }
        for (ArrayList<Long> col : intCols) {
            ArrayList<Float> fcol = new ArrayList<>(col.size());
            for (Long v : col) {
                fcol.add(v == null ? Float.NaN : v.floatValue());
            }
            out.add(fcol);
        }
        return out;
    }

    /** @deprecated Use {@link #doubleColumnsForCodecBench}. */
    public static ArrayList<ArrayList<Float>> floatColumnsForCodecBench(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols) {
        if ((tsType == TSDataType.FLOAT || tsType == TSDataType.DOUBLE)
                && floatCols != null
                && !floatCols.isEmpty()) {
            return floatCols;
        }
        if ((tsType == TSDataType.INT32 || tsType == TSDataType.INT64)
                && intCols != null
                && !intCols.isEmpty()) {
            return integersAsFloatColumns(intCols);
        }
        return new ArrayList<>();
    }

}
