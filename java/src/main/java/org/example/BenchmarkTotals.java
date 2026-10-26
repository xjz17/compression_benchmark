package org.example;

import java.util.LinkedHashMap;
import java.util.Map;

/** Accumulates size/time and round-trip loss metrics across CSV patches. */
public final class BenchmarkTotals {
    private static final class LossAcc {
        double sumErr;
        double sumRef;
    }

    private final Map<String, long[]> byAlgorithm = new LinkedHashMap<>();
    private final Map<String, LossAcc> lossByAlgorithm = new LinkedHashMap<>();

    public void merge(String algorithm, long[] result) {
        merge(algorithm, result, 0.0, 0.0);
    }

    public void merge(String algorithm, long original, long compressed, long compTime, long decompTime) {
        merge(algorithm, original, compressed, compTime, decompTime, 0.0, 0.0);
    }

    public void merge(String algorithm, long[] result, double lossAbsError, double lossAbsRef) {
        if (algorithm == null || result == null || result.length < 4 || result[0] <= 0) {
            return;
        }
        merge(
                algorithm,
                result[0],
                result[1],
                result[2],
                result[3],
                lossAbsError,
                lossAbsRef);
    }

    public void merge(
            String algorithm,
            long original,
            long compressed,
            long compTime,
            long decompTime,
            double lossAbsError,
            double lossAbsRef) {
        if (original <= 0) {
            return;
        }
        long[] acc = byAlgorithm.computeIfAbsent(algorithm, k -> new long[4]);
        acc[0] += original;
        acc[1] += compressed;
        acc[2] += compTime;
        acc[3] += decompTime;
        if (lossAbsError != 0.0 || lossAbsRef != 0.0) {
            LossAcc loss = lossByAlgorithm.computeIfAbsent(algorithm, k -> new LossAcc());
            loss.sumErr += lossAbsError;
            loss.sumRef += lossAbsRef;
        }
    }

    public Map<String, long[]> snapshot() {
        return byAlgorithm;
    }

    public int isLossy(String algorithm) {
        LossAcc loss = lossByAlgorithm.get(algorithm);
        double sumErr = loss == null ? 0.0 : loss.sumErr;
        double sumRef = loss == null ? 0.0 : loss.sumRef;
        return BenchmarkLoss.isLossyFlag(algorithm, sumErr, sumRef);
    }

    public double lossRate(String algorithm) {
        LossAcc loss = lossByAlgorithm.get(algorithm);
        if (loss == null) {
            return 0.0;
        }
        return BenchmarkLoss.lossRate(loss.sumErr, loss.sumRef);
    }

    public String lossRateCsv(String algorithm) {
        return BenchmarkLoss.lossRateCsv(lossRate(algorithm));
    }
}
