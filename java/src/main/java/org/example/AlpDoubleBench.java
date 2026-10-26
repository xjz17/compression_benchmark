package org.example;

import java.util.List;

/**
 * ALP (double/64-bit) scalar benchmark.
 *
 * <p>Matches {@link AlpFloatBench} flow, but runs on IEEE-754 doubles and int64 encoded values.
 * Integer payload is compressed via {@link Int64ColumnCodec#fastPForEncodeInt64(long[], int)}.
 */
final class AlpDoubleBench {

    private static final int MAX_EXPONENT = 10;
    private static final double MAGIC_NUMBER = 6755399441055744.0;
    private static final int EXCEPTION_POSITION_SIZE = 16;
    private static final int EXCEPTION_VALUE_SIZE = 64;
    private static final double ENCODING_UPPER_LIMIT = 9223372036854774784.0;
    private static final double ENCODING_LOWER_LIMIT = -9223372036854774784.0;

    private static final double[] FRAC_ARR = {
            1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001,
            0.000000001, 0.0000000001
    };

    private static final double[] EXP_ARR = {
            1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
            100000000.0, 1000000000.0, 10000000000.0
    };

    private static final long[] FACT_ARR = {
            1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L
    };

    private static final int MAX_FACTOR_IDX = FACT_ARR.length - 1;

    private AlpDoubleBench() {}

    private static boolean encodeRawBitsBlock(
            double[] window, int vecLen, long[] outCompCtDt, double[] lossAcc) {
        try {
            long[] rawBits = new long[vecLen];
            for (int i = 0; i < vecLen; i++) {
                rawBits[i] = Double.doubleToRawLongBits(window[i]);
            }
            long t0 = System.nanoTime();
            byte[] payload = Int64ColumnCodec.fastPForEncodeInt64(rawBits, vecLen);
            long t1 = System.nanoTime();
            long t2 = System.nanoTime();
            long[] recovered = Int64ColumnCodec.fastPForDecodeInt64(payload, vecLen);
            long t3 = System.nanoTime();
            if (recovered.length != vecLen) {
                return false;
            }
            for (int i = 0; i < vecLen; i++) {
                if (recovered[i] != rawBits[i]) {
                    return false;
                }
            }
            if (lossAcc != null) {
                double[] dec = new double[vecLen];
                for (int i = 0; i < vecLen; i++) {
                    dec[i] = Double.longBitsToDouble(recovered[i]);
                }
                BenchmarkLoss.accumulateDoubleArrayLoss(window, dec, vecLen, lossAcc);
            }
            outCompCtDt[0] += payload.length;
            outCompCtDt[1] += t1 - t0;
            outCompCtDt[2] += t3 - t2;
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean encodeRawBitsBlock(double[] window, int vecLen, long[] outCompCtDt) {
        return encodeRawBitsBlock(window, vecLen, outCompCtDt, null);
    }

    static boolean isImpossibleToEncode(double n) {
        if (!Double.isFinite(n)) {
            return true;
        }
        return n > ENCODING_UPPER_LIMIT || n < ENCODING_LOWER_LIMIT;
    }

    static long encodeValue(double value, int factorIdx, int exponentIdx) {
        double tmp = value * EXP_ARR[exponentIdx] * FRAC_ARR[factorIdx];
        if (isImpossibleToEncode(tmp)) {
            return Long.MAX_VALUE;
        }
        tmp = tmp + MAGIC_NUMBER - MAGIC_NUMBER;
        return (long) tmp;
    }

    static double decodeValue(long encoded, int factorIdx, int exponentIdx) {
        return ((double) encoded) * FACT_ARR[factorIdx] * FRAC_ARR[exponentIdx];
    }

    static int bitWidthUnsignedSpan(long minEnc, long maxEnc) {
        long delta = maxEnc - minEnc;
        if (delta == 0L) {
            return 0;
        }
        return 64 - Long.numberOfLeadingZeros(delta);
    }

    static int[] findBestFactorExponent(double[] values, int n) {
        if (n == 1) {
            double v = values[0];
            if (isImpossibleToEncode(v)) {
                return null;
            }
            for (int exp = MAX_EXPONENT; exp >= 0; exp--) {
                for (int fac = Math.min(exp, MAX_FACTOR_IDX); fac >= 0; fac--) {
                    long enc = encodeValue(v, fac, exp);
                    double dec = decodeValue(enc, fac, exp);
                    if (Double.doubleToRawLongBits(dec) == Double.doubleToRawLongBits(v)) {
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
                long minE = -1L;
                long maxE = 0L;
                boolean bad = false;
                for (int i = 0; i < n; i++) {
                    double v = values[i];
                    if (isImpossibleToEncode(v)) {
                        bad = true;
                        break;
                    }
                    long enc = encodeValue(v, fac, exp);
                    double dec = decodeValue(enc, fac, exp);
                    if (Double.doubleToRawLongBits(dec) == Double.doubleToRawLongBits(v)) {
                        nonExc++;
                        if (Long.compareUnsigned(enc, minE) < 0) {
                            minE = enc;
                        }
                        if (Long.compareUnsigned(enc, maxE) > 0) {
                            maxE = enc;
                        }
                    }
                }
                if (bad || nonExc < 2) {
                    continue;
                }
                int bits = bitWidthUnsignedSpan(minE, maxE);
                long cost =
                        (long) n * bits + (long) (n - nonExc) * (EXCEPTION_VALUE_SIZE + EXCEPTION_POSITION_SIZE);
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

    static void benchAlpDouble(List<? extends List<Double>> columns, long[] out) {
        benchAlpDouble(columns, out, null);
    }

    static void benchAlpDouble(List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long orig = 0L;
        long comp = 0L;
        long ct = 0L;
        long dt = 0L;
        try {
            for (List<Double> col : columns) {
                int colN = col.size();
                if (colN == 0) {
                    continue;
                }
                long colComp = 0L;
                long colCt = 0L;
                long colDt = 0L;
                boolean colOk = true;
                for (int off = 0; off < colN && colOk; ) {
                    int vecLen = Math.min(1024, colN - off);
                    double[] window = new double[vecLen];
                    for (int i = 0; i < vecLen; i++) {
                        Double v = col.get(off + i);
                        window[i] = v == null ? Double.NaN : v;
                    }
                    int[] fe = findBestFactorExponent(window, vecLen);
                    if (fe == null) {
                        long[] stats = new long[3];
                        if (!encodeRawBitsBlock(window, vecLen, stats, lossAcc)) {
                            colOk = false;
                            break;
                        }
                        colComp += stats[0];
                        colCt += stats[1];
                        colDt += stats[2];
                        off += vecLen;
                        continue;
                    }
                    int fac = fe[0];
                    int exp = fe[1];
                    long[] encoded = new long[vecLen];
                    int[] excPos = new int[vecLen];
                    double[] excVal = new double[vecLen];
                    int excCount = 0;
                    long placeholder = 0L;
                    boolean foundPh = false;
                    for (int i = 0; i < vecLen; i++) {
                        double v = window[i];
                        long enc = encodeValue(v, fac, exp);
                        double dec = decodeValue(enc, fac, exp);
                        if (Double.doubleToRawLongBits(dec) == Double.doubleToRawLongBits(v)) {
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
                        long[] stats = new long[3];
                        if (!encodeRawBitsBlock(window, vecLen, stats, lossAcc)) {
                            colOk = false;
                            break;
                        }
                        colComp += stats[0];
                        colCt += stats[1];
                        colDt += stats[2];
                        off += vecLen;
                        continue;
                    }
                    for (int j = 0; j < excCount; j++) {
                        encoded[excPos[j]] = placeholder;
                    }
                    boolean blockOk = true;
                    long blockComp = 0L;
                    long blockCt = 0L;
                    long blockDt = 0L;
                    try {
                        long t0 = System.nanoTime();
                        byte[] payload = Int64ColumnCodec.fastPForEncodeInt64(encoded, vecLen);
                        long t1 = System.nanoTime();
                        long t2 = System.nanoTime();
                        long[] recoveredEnc = Int64ColumnCodec.fastPForDecodeInt64(payload, vecLen);
                        long t3 = System.nanoTime();
                        if (recoveredEnc.length != vecLen) {
                            blockOk = false;
                        } else {
                            double[] roundTrip = new double[vecLen];
                            for (int i = 0; i < vecLen; i++) {
                                roundTrip[i] = decodeValue(recoveredEnc[i], fac, exp);
                            }
                            for (int j = 0; j < excCount; j++) {
                                roundTrip[excPos[j]] = excVal[j];
                            }
                            for (int i = 0; i < vecLen; i++) {
                                if (Double.doubleToRawLongBits(roundTrip[i])
                                        != Double.doubleToRawLongBits(window[i])) {
                                    blockOk = false;
                                    break;
                                }
                            }
                        }
                        if (blockOk) {
                            blockComp = payload.length + 6L + (long) excCount * (2L + 8L);
                            blockCt = t1 - t0;
                            blockDt = t3 - t2;
                            if (lossAcc != null) {
                                double[] roundTrip = new double[vecLen];
                                for (int i = 0; i < vecLen; i++) {
                                    roundTrip[i] = decodeValue(recoveredEnc[i], fac, exp);
                                }
                                for (int j = 0; j < excCount; j++) {
                                    roundTrip[excPos[j]] = excVal[j];
                                }
                                BenchmarkLoss.accumulateDoubleArrayLoss(window, roundTrip, vecLen, lossAcc);
                            }
                        }
                    } catch (Throwable ignored) {
                        blockOk = false;
                    }
                    if (!blockOk) {
                        long[] stats = new long[3];
                        if (!encodeRawBitsBlock(window, vecLen, stats, lossAcc)) {
                            colOk = false;
                            break;
                        }
                        colComp += stats[0];
                        colCt += stats[1];
                        colDt += stats[2];
                        off += vecLen;
                        continue;
                    }
                    colComp += blockComp;
                    colCt += blockCt;
                    colDt += blockDt;
                    off += vecLen;
                }
                if (!colOk) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(colN, true);
                comp += colComp;
                ct += colCt;
                dt += colDt;
            }
        } catch (Throwable ignored) {
            out[0] = 0L;
            out[1] = 0L;
            out[2] = 0L;
            out[3] = 0L;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }
}
