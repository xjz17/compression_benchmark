package org.example;

/** Dataset input policy: omit NaN values, retaining infinity and signed zero. */
public final class ExperimentDataPolicy {
    private ExperimentDataPolicy() {}

    public static boolean keepNumericValue(double value) {
        return !Double.isNaN(value);
    }

    public static boolean isNaNCell(String value) {
        if (value == null) return false;
        String cell = value.trim();
        return cell.equalsIgnoreCase("nan") || cell.equalsIgnoreCase("+nan") || cell.equalsIgnoreCase("-nan");
    }
}
