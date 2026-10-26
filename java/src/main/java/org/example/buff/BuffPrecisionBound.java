package org.example.buff;

/**
 * Java port of BUFF's bounded-double core from upstream
 * {@code external/compression/buff/database/src/methods/prec_double.rs} ({@code PrecisionBound}).
 *
 * <p>This is the bitwise &quot;precision bound&quot; and component split used by Tranway1/buff
 * (VLDB 2021), not the subprocess wrapper in {@link org.example.ResearchBuffCliBench}.
 */
public final class BuffPrecisionBound {

    /** IEEE754 exponent field mask. */
    public static final long EXP_MASK = 0x7FF0000000000000L;
    /** Sign bit mask (high bit). */
    public static final long FIRST_ONE = 0x8000000000000000L;
    public static final long NEG_ONE = 0xFFFFFFFFFFFFFFFFL;

    private long position;
    private final double precision;
    private final int precisionExp;
    private long intLength;
    private long decimalLength;

    /**
     * Mirrors Rust {@code get_precision_bound}: builds {@code 0.000...049} with {@code precision}
     * zeros after the decimal point, then parses as {@code double}.
     */
    public static double getPrecisionBound(int precision) {
        StringBuilder sb = new StringBuilder("0.");
        for (int i = 0; i < precision; i++) {
            sb.append('0');
        }
        sb.append("49");
        return Double.parseDouble(sb.toString());
    }

    public BuffPrecisionBound(double precDelta) {
        this.precision = precDelta;
        long xu = Double.doubleToRawLongBits(precDelta);
        this.precisionExp = (int) ((xu & EXP_MASK) >> 52) - 1023;
        this.position = 0L;
        this.intLength = 0L;
        this.decimalLength = 0L;
    }

    /**
     * Find a bounded double near {@code orig} subject to {@link #isBounded(double, double)}.
     * Port of {@code PrecisionBound::precision_bound}.
     */
    public double precisionBound(double orig) {
        long mask = ~0L;
        double pre = 0.0;
        double cur;
        long origu = Double.doubleToRawLongBits(orig);
        long curu;
        curu = (origu & (mask << position)) | (1L << position);
        cur = Double.longBitsToDouble(curu);
        pre = cur;
        boolean bounded = isBounded(orig, cur);
        if (bounded) {
            while (bounded) {
                if (position == 52) {
                    return pre;
                }
                position += 1;
                curu = (origu & (mask << position)) | (1L << position);
                cur = Double.longBitsToDouble(curu);
                if (!isBounded(orig, cur)) {
                    bounded = false;
                    break;
                }
                pre = cur;
            }
        } else {
            while (!bounded) {
                if (position == 0) {
                    break;
                }
                position -= 1;
                curu = (origu & (mask << position)) | (1L << position);
                cur = Double.longBitsToDouble(curu);
                if (isBounded(orig, cur)) {
                    bounded = true;
                    pre = cur;
                    break;
                }
            }
        }
        return pre;
    }

    /** Update max integer / decimal bit lengths for a bounded value. Port of {@code cal_length}. */
    public void calLength(double x) {
        long xu = Double.doubleToRawLongBits(x);
        int trailingZeros = Long.numberOfTrailingZeros(xu);
        int exp = (int) ((xu & EXP_MASK) >> 52) - 1023;
        long decLength = 0;
        if (52 <= trailingZeros) {
            if (exp < 0) {
                decLength = -exp;
                if (exp < precisionExp) {
                    decLength = 0;
                }
            }
        } else if ((52 - trailingZeros) > exp) {
            decLength = (52 - trailingZeros) - exp;
        }
        if (exp >= 0) {
            if ((exp + 1L) > intLength) {
                intLength = (exp + 1L);
            }
        }
        if (decLength > decimalLength) {
            decimalLength = decLength;
        }
    }

    public long getIntLength() {
        return intLength;
    }

    public long getDecimalLength() {
        return decimalLength;
    }

    public long[] getLength() {
        return new long[] {intLength, decimalLength};
    }

    public void setLength(long ilen, long dlen) {
        this.intLength = ilen;
        this.decimalLength = dlen;
    }

    /**
     * Split a bounded double into integer and fractional bit fields. Port of
     * {@code fetch_components}.
     */
    public long[] fetchComponents(double bd) {
        long bdu = Double.doubleToRawLongBits(bd);
        int exp = (int) ((bdu & EXP_MASK) >> 52) - 1023;
        long sign = bdu & FIRST_ONE;
        long intPart = 0;
        long decPart = 0;
        if (exp >= 0) {
            decPart = bdu << (12 + exp);
            intPart = ((bdu << 11) | FIRST_ONE) >>> (63 - exp);
            if (sign != 0) {
                intPart = ~intPart;
                decPart = ~decPart + 1;
            }
        } else if (exp < precisionExp) {
            decPart = 0;
            if (sign != 0) {
                intPart = NEG_ONE;
                decPart = ~decPart;
            }
        } else {
            decPart = ((bdu << 11) | FIRST_ONE) >>> (-exp - 1);
            if (sign != 0) {
                intPart = NEG_ONE;
                decPart = ~decPart;
            }
        }
        long signedInt = intPart;
        long shift = 64L - decimalLength;
        long decOut = shift > 0 ? (decPart >>> shift) : decPart;
        return new long[] {signedInt, decOut};
    }

    public boolean isBounded(double a, double b) {
        double delta = a - b;
        return Math.abs(delta) < precision;
    }
}
