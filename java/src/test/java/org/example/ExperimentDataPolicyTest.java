package org.example;

/** Standalone regression fixture; no JUnit or codec dependencies needed. */
public final class ExperimentDataPolicyTest {
    public static void main(String[] args) {
        if (ExperimentDataPolicy.keepNumericValue(Double.NaN)) throw new AssertionError("NaN");
        if (!ExperimentDataPolicy.keepNumericValue(Double.POSITIVE_INFINITY)) throw new AssertionError("Inf");
        if (!ExperimentDataPolicy.keepNumericValue(Double.NEGATIVE_INFINITY)) throw new AssertionError("-Inf");
        if (!ExperimentDataPolicy.keepNumericValue(-0.0) || Double.doubleToRawLongBits(-0.0) != Long.MIN_VALUE)
            throw new AssertionError("signed zero");
        if (!ExperimentDataPolicy.isNaNCell(" -nAn ")) throw new AssertionError("NaN cell");
        if (Long.parseLong("9007199254740993") != 9007199254740993L) throw new AssertionError("integer precision");
    }
}
