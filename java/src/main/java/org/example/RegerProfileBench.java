package org.example;

public final class RegerProfileBench {
    private RegerProfileBench() {}

    public static void main(String[] args) {
        final int rows = 200_000;
        final int columns = 3;
        long[] times = new long[rows];
        long[][] values = new long[columns][rows];
        for (int i = 0; i < rows; i++) {
            times[i] = 1_700_000_000_000L + (long) i * 1_000L;
            values[0][i] = 10_000L + (long) i * 3L + i % 5;
            values[1][i] = -50_000L + (long) i * 7L + i % 3;
            values[2][i] = 800L + i / 16L;
        }

        byte[] encoded = RegerCodec.encodeInt64Columns(values, times);
        long[][] decoded = RegerCodec.decodeInt64Columns(encoded);
        for (int c = 0; c < columns; c++) {
            for (int i = 0; i < rows; i++) {
                if (decoded[c][i] != values[c][i]) {
                    throw new AssertionError("REGER profile round trip failed");
                }
            }
        }

        long[] duplicateUnsortedTimes = {5, 2, 2, 9, 1, 5, 3, 3};
        long[][] reorderedValues = {
            {90, 10, 80, 20, 70, 30, 60, 40},
            {901, 101, 801, 201, 701, 301, 601, 401},
        };
        byte[] reorderedPayload =
                RegerCodec.encodeInt64Columns(reorderedValues, duplicateUnsortedTimes);
        long[][] reorderedDecoded = RegerCodec.decodeInt64Columns(reorderedPayload);
        for (int c = 0; c < reorderedValues.length; c++) {
            for (int i = 0; i < duplicateUnsortedTimes.length; i++) {
                if (reorderedDecoded[c][i] != reorderedValues[c][i]) {
                    throw new AssertionError("REGER permutation round trip failed");
                }
            }
        }

        long bestNanos = Long.MAX_VALUE;
        for (int run = 0; run < 5; run++) {
            long start = System.nanoTime();
            encoded = RegerCodec.encodeInt64Columns(values, times);
            bestNanos = Math.min(bestNanos, System.nanoTime() - start);
        }
        double points = (double) rows * columns;
        double ratio = points * Long.BYTES / encoded.length;
        String profile = System.getenv("WEB_COMPRESSION_REGER_PROFILE");
        if (profile == null || profile.isBlank()) {
            profile = "balanced";
        }
        System.out.printf("%s, ns/point=%.1f, ratio=%.3f, bytes=%d%n",
                profile, bestNanos / points, ratio, encoded.length);
    }
}
