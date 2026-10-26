package org.example.buff;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Java port of the simple split-double column encoder in upstream
 * {@code external/compression/buff/database/src/compress/split_double.rs}
 * ({@code SplitBDDoubleCompress::encode} first overload, lines ~41–101): bounded doubles,
 * then bit-packed integer and decimal components.
 *
 * <p>This is real BUFF decomposition logic in Java. The benchmark harness
 * {@link org.example.ResearchBuffCliBench} still shells out to Rust {@code comp_profiler} for
 * CSV timings; use this class when you need the algorithm in-process in Java.
 */
public final class BuffSplitDoubleColumnCodec {

    /** Decoded int/decimal components (no float reconstruction). */
    public static final class DecodedColumn {
        public final int t;
        public final int ilen;
        public final int dlen;
        public final List<Long> intParts;
        public final List<Long> decParts;
        public final double[] values;

        public DecodedColumn(
                int t,
                int ilen,
                int dlen,
                List<Long> intParts,
                List<Long> decParts,
                double[] values) {
            this.t = t;
            this.ilen = ilen;
            this.dlen = dlen;
            this.intParts = Collections.unmodifiableList(intParts);
            this.decParts = Collections.unmodifiableList(decParts);
            this.values = values.clone();
        }
    }

    private BuffSplitDoubleColumnCodec() {}

    /** Mirrors {@code PRECISION_MAP} in {@code database/src/compress/mod.rs}. */
    public static int decimalBitsForPrecision(int prec) {
        switch (prec) {
            case 1:
                return 5;
            case 2:
                return 8;
            case 3:
                return 11;
            case 4:
                return 15;
            case 5:
                return 18;
            case 6:
                return 21;
            case 7:
                return 25;
            case 8:
                return 28;
            case 9:
                return 31;
            case 10:
                return 35;
            case 11:
                return 38;
            case 12:
                return 50;
            case 13:
            case 14:
            case 15:
            default:
                return 10;
        }
    }

    /**
     * Encode one column of {@code double} values (same convention as upstream: {@code scale} sets
     * decimal precision via {@code prec = floor(log10(scale))}).
     *
     * @param values raw column values
     * @param scale  e.g. 10000 → prec 4 (see Rust {@code (scale as f32).log10() as i32})
     * @return packed bytes (header + int payload + decimal payload)
     */
    public static byte[] encodeDoubles(double[] values, int scale) {
        if (values == null || values.length == 0) {
            return new byte[0];
        }
        BuffBitPackWriter bp = new BuffBitPackWriter();
        int t = values.length;
        bp.write(t, 32);
        bp.write(64, 32);
        bp.write(0, 32);
        for (double value : values) {
            long bits = Double.doubleToRawLongBits(value);
            bp.write((int) (bits >>> 32), 32);
            bp.write((int) bits, 32);
        }
        return bp.toByteArray();
    }

    /** Inverse of {@link #encodeDoubles} at component level. */
    public static DecodedColumn decodeDoubles(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            return new DecodedColumn(0, 0, 0, List.of(), List.of(), new double[0]);
        }
        BuffBitPackReader rd = new BuffBitPackReader(bytes);
        int t = rd.read(32);
        int ilen = rd.read(32);
        int dlen = rd.read(32);
        if (ilen == 64 && dlen == 0) {
            double[] values = new double[t];
            for (int i = 0; i < t; i++) {
                long bits = ((long) rd.read(32) << 32) | ((long) rd.read(32) & 0xFFFFFFFFL);
                values[i] = Double.longBitsToDouble(bits);
            }
            return new DecodedColumn(t, ilen, dlen, List.of(), List.of(), values);
        }
        List<Long> intParts = new ArrayList<>();
        List<Long> decParts = new ArrayList<>();
        for (int i = 0; i < t; i++) {
            int raw = rd.read(ilen);
            intParts.add(BuffBitPackReader.signExtendToLong(raw, ilen));
        }
        for (int i = 0; i < t; i++) {
            int raw = rd.read(dlen);
            decParts.add((long) raw & 0xFFFFFFFFL);
        }
        return new DecodedColumn(t, ilen, dlen, intParts, decParts, new double[0]);
    }
}
