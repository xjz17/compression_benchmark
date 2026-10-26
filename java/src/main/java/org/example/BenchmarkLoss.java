package org.example;

import java.util.List;
import java.util.Locale;

/** Round-trip loss metrics for benchmark CSV reports (``isLossy``, ``lossRate``). */
public final class BenchmarkLoss {

    private static final double LOSS_RATE_EPS = 1e-15;

    private static final String[] KNOWN_LOSSY_PREFIXES = {
        "sz_",
        "sz3_",
        "serf_",
        "simpiece_",
        "ext_zfp_tol_",
        "machete_rel_",
        "machete_float_rel_",
        "most_minimal",
        "most_float_minimal",
    };

    private BenchmarkLoss() {
    }

    public static boolean isKnownLossyAlgorithm(String algorithm) {
        if (algorithm == null) {
            return false;
        }
        String key = algorithm.trim().toLowerCase();
        if (key.isEmpty()) {
            return false;
        }
        for (String prefix : KNOWN_LOSSY_PREFIXES) {
            if (key.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    public static void accumulateByteLoss(byte[] original, byte[] decoded, double[] acc) {
        if (acc == null || acc.length < 2 || original == null) {
            return;
        }
        int ref = original.length;
        int err = 0;
        if (decoded == null) {
            err = ref;
        } else {
            int n = Math.min(original.length, decoded.length);
            for (int i = 0; i < n; i++) {
                if (original[i] != decoded[i]) {
                    err++;
                }
            }
            err += Math.abs(original.length - decoded.length);
        }
        acc[0] += err;
        acc[1] += ref;
    }

    public static void accumulateLongLoss(List<Long> original, long[] decoded, int decodedCount, double[] acc) {
        if (acc == null || acc.length < 2 || original == null || decoded == null) {
            return;
        }
        int n = Math.min(original.size(), decodedCount);
        double err = 0;
        for (int i = 0; i < n; i++) {
            long o = original.get(i);
            long d = decoded[i];
            if (o != d) err++;
        }
        if (original.size() != decodedCount) {
            err += Math.abs(original.size() - decodedCount);
        }
        acc[0] += err;
        acc[1] += Math.max(original.size(), decodedCount);
    }

    public static void accumulateDoubleLoss(
            List<Double> original, double[] decoded, int decodedCount, double[] acc) {
        if (acc == null || acc.length < 2 || original == null || decoded == null) {
            return;
        }
        int n = Math.min(original.size(), decodedCount);
        double err = 0;
        for (int i = 0; i < n; i++) {
            double o = original.get(i);
            double d = decoded[i];
            if (Double.doubleToRawLongBits(o) != Double.doubleToRawLongBits(d)) err++;
        }
        if (original.size() != decodedCount) {
            err += Math.abs(original.size() - decodedCount);
        }
        acc[0] += err;
        acc[1] += Math.max(original.size(), decodedCount);
    }

    public static void accumulateLongListLoss(List<Long> orig, List<Long> dec, double[] acc) {
        if (acc == null || acc.length < 2 || orig == null || dec == null) {
            return;
        }
        int n = Math.min(orig.size(), dec.size());
        double err = 0;
        for (int i = 0; i < n; i++) {
            long o = orig.get(i) == null ? 0L : orig.get(i);
            long d = dec.get(i) == null ? 0L : dec.get(i);
            if (o != d) err++;
        }
        if (orig.size() != dec.size()) {
            err += Math.abs(orig.size() - dec.size());
        }
        acc[0] += err;
        acc[1] += Math.max(orig.size(), dec.size());
    }

    public static void accumulateDoubleListLoss(List<Double> orig, List<Double> dec, double[] acc) {
        if (acc == null || acc.length < 2 || orig == null || dec == null) {
            return;
        }
        int n = Math.min(orig.size(), dec.size());
        double err = 0;
        for (int i = 0; i < n; i++) {
            Double ov = orig.get(i);
            Double dv = dec.get(i);
            double o = ov == null ? Double.NaN : ov;
            double d = dv == null ? Double.NaN : dv;
            if (Double.doubleToRawLongBits(o) != Double.doubleToRawLongBits(d)) err++;
        }
        if (orig.size() != dec.size()) {
            err += Math.abs(orig.size() - dec.size());
        }
        acc[0] += err;
        acc[1] += Math.max(orig.size(), dec.size());
    }

    public static void accumulateLongArrayLoss(long[] orig, long[] dec, int n, double[] acc) {
        if (acc == null || acc.length < 2 || orig == null || dec == null) {
            return;
        }
        int count = Math.min(n, Math.min(orig.length, dec.length));
        double err = 0;
        for (int i = 0; i < count; i++) {
            if (orig[i] != dec[i]) err++;
        }
        if (n != count) {
            err += Math.abs(n - count);
        }
        acc[0] += err;
        acc[1] += Math.max(n, count);
    }

    public static void accumulateDoubleArrayLoss(double[] orig, double[] dec, int n, double[] acc) {
        if (acc == null || acc.length < 2 || orig == null || dec == null) {
            return;
        }
        int count = Math.min(n, Math.min(orig.length, dec.length));
        double err = 0;
        for (int i = 0; i < count; i++) {
            if (Double.doubleToRawLongBits(orig[i]) != Double.doubleToRawLongBits(dec[i])) err++;
        }
        if (n != count) {
            err += Math.abs(n - count);
        }
        acc[0] += err;
        acc[1] += Math.max(n, count);
    }

    public static double lossRate(double sumAbsError, double sumAbsRef) {
        if (sumAbsRef <= 0.0) {
            return sumAbsError > 0.0 ? 1.0 : 0.0;
        }
        return sumAbsError / sumAbsRef;
    }

    public static int isLossyFlag(String algorithm, double sumAbsError, double sumAbsRef) {
        if (isKnownLossyAlgorithm(algorithm)) {
            return 1;
        }
        return lossRate(sumAbsError, sumAbsRef) > LOSS_RATE_EPS ? 1 : 0;
    }

    /** Fixed decimal form for CSV (never scientific notation). */
    public static String lossRateCsv(double rate) {
        if (rate == 0.0 || !Double.isFinite(rate)) {
            return "0";
        }
        String s = String.format(Locale.US, "%.15f", rate);
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == '0') {
            end--;
        }
        if (end > 0 && s.charAt(end - 1) == '.') {
            end--;
        }
        return end == 0 ? "0" : s.substring(0, end);
    }
}
