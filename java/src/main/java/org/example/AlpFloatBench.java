package org.example;

import me.lemire.integercompression.Composition;
import me.lemire.integercompression.FastPFOR;
import me.lemire.integercompression.IntWrapper;
import me.lemire.integercompression.IntegerCODEC;
import me.lemire.integercompression.VariableByte;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ALP (float) scalar benchmark aligned with cwida/ALP {@code encoder<float>} / {@code decoder<float>}:
 * factor–exponent search, exception list, integers compressed with FastPFOR+VariableByte per vector.
 */
final class AlpFloatBench {

    private static final int MAX_EXPONENT = 10;
    private static final float MAGIC_NUMBER = 12582912.0f;
    private static final int EXCEPTION_SIZE = 32;
    private static final int EXCEPTION_POSITION_SIZE = 16;
    private static final double ENCODING_UPPER_LIMIT = 9223372036854774784.0;
    private static final double ENCODING_LOWER_LIMIT = -9223372036854774784.0;

    private static final float[] FRAC_ARR = {
            1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f, 0.00000001f,
            0.000000001f, 0.0000000001f
    };

    private static final float[] EXP_ARR = {
            1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 1000000.0f, 10000000.0f,
            100000000.0f, 1000000000.0f, 10000000000.0f
    };

    private static final int[] FACT_ARR = {
            1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
    };

    /** Safe index for {@link #decodeValue}: {@code FACT_ARR} has no index 10 (C++ may UB on 10). */
    private static final int MAX_FACTOR_IDX = FACT_ARR.length - 1;

    private AlpFloatBench() {}

    static boolean isImpossibleToEncode(float n) {
        if (!Float.isFinite(n)) {
            return true;
        }
        double d = n;
        if (d > ENCODING_UPPER_LIMIT || d < ENCODING_LOWER_LIMIT) {
            return true;
        }
        return false;
    }

    static int encodeValue(float value, int factorIdx, int exponentIdx) {
        float tmp = value * EXP_ARR[exponentIdx] * FRAC_ARR[factorIdx];
        if (isImpossibleToEncode(tmp)) {
            return Integer.MAX_VALUE;
        }
        tmp = tmp + MAGIC_NUMBER - MAGIC_NUMBER;
        return (int) tmp;
    }

    static float decodeValue(int encoded, int factorIdx, int exponentIdx) {
        return (float) ((double) encoded * FACT_ARR[factorIdx] * FRAC_ARR[exponentIdx]);
    }

    static int bitWidthUnsignedSpan(int minEnc, int maxEnc) {
        long delta = Integer.toUnsignedLong(maxEnc) - Integer.toUnsignedLong(minEnc);
        if (delta == 0) {
            return 0;
        }
        return 64 - Long.numberOfLeadingZeros(delta);
    }

    /**
     * Best (factor, exponent) for one vector; {@code null} if no pair with at least two exact encodings.
     */
    static int[] findBestFactorExponent(float[] values, int n) {
        if (n == 1) {
            float v = values[0];
            if (isImpossibleToEncode(v)) {
                return null;
            }
            for (int exp = MAX_EXPONENT; exp >= 0; exp--) {
                for (int fac = Math.min(exp, MAX_FACTOR_IDX); fac >= 0; fac--) {
                    int enc = encodeValue(v, fac, exp);
                    float dec = decodeValue(enc, fac, exp);
                    if (Float.floatToRawIntBits(dec) == Float.floatToRawIntBits(v)) {
                        return new int[] {fac, exp};
                    }
                }
            }
            return null;
        }
        long bestCost = Long.MAX_VALUE;
        int bestFac = 0;
        int bestExp = 0;
        for (int exp = MAX_EXPONENT; exp >= 0; exp--) {
            for (int fac = Math.min(exp, MAX_FACTOR_IDX); fac >= 0; fac--) {
                int nonExc = 0;
                int minE = Integer.MAX_VALUE;
                int maxE = Integer.MIN_VALUE;
                boolean bad = false;
                for (int i = 0; i < n; i++) {
                    float v = values[i];
                    if (isImpossibleToEncode(v)) {
                        bad = true;
                        break;
                    }
                    int enc = encodeValue(v, fac, exp);
                    float dec = decodeValue(enc, fac, exp);
                    if (Float.floatToRawIntBits(dec) == Float.floatToRawIntBits(v)) {
                        nonExc++;
                        if (enc < minE) {
                            minE = enc;
                        }
                        if (enc > maxE) {
                            maxE = enc;
                        }
                    }
                }
                if (bad || nonExc < 2) {
                    continue;
                }
                int bits = bitWidthUnsignedSpan(minE, maxE);
                long cost = (long) n * bits + (long) (n - nonExc) * (EXCEPTION_SIZE + EXCEPTION_POSITION_SIZE);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestFac = fac;
                    bestExp = exp;
                }
            }
        }
        if (bestCost == Long.MAX_VALUE) {
            return null;
        }
        return new int[] {bestFac, bestExp};
    }

    static void benchAlpFloat(List<? extends List<Float>> columns, long[] out) {
        benchAlpFloat(columns, out, null);
    }

    static void benchAlpFloat(List<? extends List<Float>> columns, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        try {
            for (List<Float> col : columns) {
                int colN = col.size();
                if (colN == 0) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(colN, true);
                long colComp = 0;
                long colCt = 0;
                long colDt = 0;
                boolean colOk = true;
                for (int off = 0; off < colN && colOk; ) {
                    int vecLen = Math.min(1024, colN - off);
                    float[] window = new float[vecLen];
                    for (int i = 0; i < vecLen; i++) {
                        window[i] = col.get(off + i);
                    }
                    int[] fe = findBestFactorExponent(window, vecLen);
                    if (fe == null) {
                        colOk = false;
                        break;
                    }
                    int fac = fe[0];
                    int exp = fe[1];
                    int[] encoded = new int[vecLen];
                    int[] excPos = new int[vecLen];
                    float[] excVal = new float[vecLen];
                    int excCount = 0;
                    int placeholder = 0;
                    boolean foundPh = false;
                    for (int i = 0; i < vecLen; i++) {
                        float v = window[i];
                        int enc = encodeValue(v, fac, exp);
                        float dec = decodeValue(enc, fac, exp);
                        if (Float.floatToRawIntBits(dec) == Float.floatToRawIntBits(v)) {
                            encoded[i] = enc;
                            if (!foundPh) {
                                placeholder = enc;
                                foundPh = true;
                            }
                        } else {
                            excPos[excCount] = i;
                            excVal[excCount] = v;
                            excCount++;
                        }
                    }
                    if (!foundPh) {
                        colOk = false;
                        break;
                    }
                    for (int j = 0; j < excCount; j++) {
                        encoded[excPos[j]] = placeholder;
                    }
                    int[] buf = new int[Math.max(vecLen * 2, vecLen + 4096)];
                    IntWrapper inOff = new IntWrapper(0);
                    IntWrapper outOff = new IntWrapper(0);
                    long t0 = System.nanoTime();
                    codec.compress(encoded, inOff, vecLen, buf, outOff);
                    long t1 = System.nanoTime();
                    int[] compressed = Arrays.copyOf(buf, outOff.get());
                    int[] recoveredEnc = new int[vecLen];
                    IntWrapper uIn = new IntWrapper(0);
                    IntWrapper uOut = new IntWrapper(0);
                    long t2 = System.nanoTime();
                    codec.uncompress(compressed, uIn, compressed.length, recoveredEnc, uOut);
                    long t3 = System.nanoTime();
                    if (uOut.get() != vecLen) {
                        colOk = false;
                        break;
                    }
                    float[] roundTrip = new float[vecLen];
                    for (int i = 0; i < vecLen; i++) {
                        roundTrip[i] = decodeValue(recoveredEnc[i], fac, exp);
                    }
                    for (int j = 0; j < excCount; j++) {
                        roundTrip[excPos[j]] = excVal[j];
                    }
                    for (int i = 0; i < vecLen; i++) {
                        if (Float.floatToRawIntBits(roundTrip[i])
                                != Float.floatToRawIntBits(window[i])) {
                            colOk = false;
                            break;
                        }
                    }
                    if (!colOk) {
                        break;
                    }
                    if (lossAcc != null) {
                        List<Double> origVals = new ArrayList<>(vecLen);
                        List<Double> decVals = new ArrayList<>(vecLen);
                        for (int i = 0; i < vecLen; i++) {
                            origVals.add((double) window[i]);
                            decVals.add((double) roundTrip[i]);
                        }
                        BenchmarkLoss.accumulateDoubleListLoss(origVals, decVals, lossAcc);
                    }
                    colComp += (long) compressed.length * 4L + 6L + (long) excCount * (2L + 4L);
                    colCt += t1 - t0;
                    colDt += t3 - t2;
                    off += vecLen;
                }
                if (!colOk) {
                    continue;
                }
                comp += colComp;
                ct += colCt;
                dt += colDt;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }
}
