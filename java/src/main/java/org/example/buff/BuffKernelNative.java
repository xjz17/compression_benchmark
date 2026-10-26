package org.example.buff;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Companion helpers for the in-process BUFF kernel bench row ({@code buff_float_kernel_native}): column
 * aggregates and a minimal sorted-unique id set. SIMD on Intel / Apple Silicon is implemented in C++
 * ({@code buff_simd_native.cpp}); HotSpot typically auto-vectorizes these scalar loops on both platforms.
 */
public final class BuffKernelNative {

    private BuffKernelNative() {}

    /** Returns {@code [sum, min, max]}; empty input yields zeros and +/- infinity for min/max per convention. */
    public static double[] columnStats(double[] data) {
        if (data == null || data.length == 0) {
            return new double[] {0.0, 0.0, 0.0};
        }
        double sum = 0.0;
        double min = data[0];
        double max = data[0];
        for (double v : data) {
            sum += v;
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
        return new double[] {sum, min, max};
    }

    public static long countGt(double[] data, double threshold) {
        if (data == null) {
            return 0;
        }
        long c = 0;
        for (double v : data) {
            if (v > threshold) {
                c++;
            }
        }
        return c;
    }

    /** Minimal sorted unique {@code uint32} set (filter bookkeeping), aligned with {@code BuffRoaringLite} in C++. */
    public static final class RoaringLite {
        private final List<Integer> ids = new ArrayList<>();

        public void add(int x) {
            int idx = Collections.binarySearch(ids, x);
            if (idx < 0) {
                ids.add(-idx - 1, x);
            }
        }

        public int cardinality() {
            return ids.size();
        }
    }
}
