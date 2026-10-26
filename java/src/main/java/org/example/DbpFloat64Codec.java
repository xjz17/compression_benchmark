package org.example;

import java.io.ByteArrayOutputStream;
import java.util.List;

/**
 * Self-contained float64 DBP frame.
 *
 * <p>The integer body is the project's first-value plus modulo-2^64 delta,
 * ZigZag, and unsigned LEB128 stream.  The DBF1 header records whether the
 * integer lanes are an exactly reversible decimal scaling or the original
 * IEEE-754 words.</p>
 */
public final class DbpFloat64Codec {
    public static final int MAX_BLOCK_VALUES = 10_000;
    public static final int HEADER_SIZE = 10;
    public static final int MODE_SCALED = 0;
    public static final int MODE_IEEE_BITS = 1;
    public static final int RAW_SCALE = 255;
    public static final int MAX_SCALE = 18;

    private static final byte[] MAGIC64 = {'D', 'B', 'F', '1'};
    private static final byte[] MAGIC32 = {'D', 'B', 'F', '2'};
    private static final double POSITIVE_I64_LIMIT = 0x1.0p63;
    private static final double NEGATIVE_I64_LIMIT = -0x1.0p63;

    private DbpFloat64Codec() {}

    public static byte[] encodeFrame(List<? extends Double> values, int metadataMaxPoint) {
        if (values == null) throw new IllegalArgumentException("DBF1 values are null");
        double[] primitive = new double[values.size()];
        for (int i = 0; i < primitive.length; i++) {
            Double value = values.get(i);
            if (value == null) throw new IllegalArgumentException("DBF1 null value at " + i);
            primitive[i] = value;
        }
        return encodeFrame(primitive, metadataMaxPoint);
    }

    public static byte[] encodeFrame(double[] values, int metadataMaxPoint) {
        if (values == null) throw new IllegalArgumentException("DBF1 values are null");
        if (values.length > MAX_BLOCK_VALUES) {
            throw new IllegalArgumentException("DBF1 block exceeds 10000 values");
        }

        if (values.length == 0) {
            return header(MAGIC64, MODE_IEEE_BITS, RAW_SCALE, 0).toByteArray();
        }

        long[] codes = new long[values.length];
        int chosenScale = -1;
        int upper = metadataMaxPoint < 0 ? MAX_SCALE : Math.min(metadataMaxPoint, MAX_SCALE);
        for (int scale = upper; scale >= 0; scale--) {
            if (tryScaleExactly(values, scale, codes)) {
                chosenScale = scale;
                break;
            }
        }

        final int mode;
        final int scale;
        if (chosenScale >= 0) {
            mode = MODE_SCALED;
            scale = chosenScale;
        } else {
            mode = MODE_IEEE_BITS;
            scale = RAW_SCALE;
            for (int i = 0; i < values.length; i++) {
                codes[i] = Double.doubleToRawLongBits(values[i]);
            }
        }

        ByteArrayOutputStream out = header(MAGIC64, mode, scale, values.length);
        writeLongLE(out, codes[0]);
        long previous = codes[0];
        for (int i = 1; i < codes.length; i++) {
            long current = codes[i];
            long delta = current - previous;
            previous = current;
            writeUnsignedLeb128(out, zigzag(delta));
        }
        return out.toByteArray();
    }

    public static double[] decodeFrame(byte[] frame) {
        DecodedCodes decoded = decodeCodes(frame, MAGIC64);
        double[] values = new double[decoded.codes.length];
        if (decoded.header.mode == MODE_IEEE_BITS) {
            for (int i = 0; i < decoded.codes.length; i++) {
                values[i] = Double.longBitsToDouble(decoded.codes[i]);
            }
        } else {
            double power = powerOfTen(decoded.header.scale);
            for (int i = 0; i < decoded.codes.length; i++) {
                values[i] = ((double) decoded.codes[i]) / power;
            }
        }
        return values;
    }

    /** Encode one DBF2 float32 block from original, noncanonicalized IEEE-754 words. */
    public static byte[] encodeFrame32(int[] rawWords, int metadataMaxPoint) {
        if (rawWords == null) throw new IllegalArgumentException("DBF2 words are null");
        if (rawWords.length > MAX_BLOCK_VALUES) {
            throw new IllegalArgumentException("DBF2 block exceeds 10000 values");
        }
        if (rawWords.length == 0) {
            return header(MAGIC32, MODE_IEEE_BITS, RAW_SCALE, 0).toByteArray();
        }

        long[] codes = new long[rawWords.length];
        int chosenScale = -1;
        int upper = metadataMaxPoint < 0 ? MAX_SCALE : Math.min(metadataMaxPoint, MAX_SCALE);
        for (int scale = upper; scale >= 0; scale--) {
            if (tryScale32Exactly(rawWords, scale, codes)) {
                chosenScale = scale;
                break;
            }
        }

        final int mode;
        final int scale;
        if (chosenScale >= 0) {
            mode = MODE_SCALED;
            scale = chosenScale;
        } else {
            mode = MODE_IEEE_BITS;
            scale = RAW_SCALE;
            for (int i = 0; i < rawWords.length; i++) codes[i] = Integer.toUnsignedLong(rawWords[i]);
        }
        return encodeCodes(MAGIC32, mode, scale, codes);
    }

    /** Decode one DBF2 frame to its original, noncanonicalized IEEE-754 words. */
    public static int[] decodeFrame32(byte[] frame) {
        DecodedCodes decoded = decodeCodes(frame, MAGIC32);
        int[] words = new int[decoded.codes.length];
        if (decoded.header.mode == MODE_IEEE_BITS) {
            for (int i = 0; i < decoded.codes.length; i++) {
                if ((decoded.codes[i] & 0xffffffff00000000L) != 0) {
                    throw new IllegalArgumentException("DBF2 raw word exceeds uint32");
                }
                words[i] = (int) decoded.codes[i];
            }
        } else {
            double power = powerOfTen(decoded.header.scale);
            for (int i = 0; i < decoded.codes.length; i++) {
                float value = (float) (((double) decoded.codes[i]) / power);
                words[i] = Float.floatToRawIntBits(value);
            }
        }
        return words;
    }

    public static boolean isFloat32Frame(byte[] frame) {
        return hasMagic(frame, MAGIC32);
    }

    public static int modeOf(byte[] frame) {
        return parseAnyHeader(frame).mode;
    }

    public static int scaleOf(byte[] frame) {
        return parseAnyHeader(frame).scale;
    }

    public static int countOf(byte[] frame) {
        return parseAnyHeader(frame).count;
    }

    private static boolean tryScaleExactly(double[] values, int scale, long[] codes) {
        double power = powerOfTen(scale);
        for (int i = 0; i < values.length; i++) {
            double value = values[i];
            if (!Double.isFinite(value)) return false;
            double product = value * power;
            if (!Double.isFinite(product)) return false;
            double rounded = Math.rint(product);
            if (rounded < NEGATIVE_I64_LIMIT || rounded >= POSITIVE_I64_LIMIT) return false;
            long code = (long) rounded;
            double recovered = ((double) code) / power;
            if (Double.doubleToRawLongBits(recovered) != Double.doubleToRawLongBits(value)) {
                return false;
            }
            codes[i] = code;
        }
        return true;
    }

    private static boolean tryScale32Exactly(int[] rawWords, int scale, long[] codes) {
        double power = powerOfTen(scale);
        for (int i = 0; i < rawWords.length; i++) {
            float original = Float.intBitsToFloat(rawWords[i]);
            if (!Float.isFinite(original)) return false;
            double product = ((double) original) * power;
            if (!Double.isFinite(product)) return false;
            double rounded = Math.rint(product);
            if (rounded < NEGATIVE_I64_LIMIT || rounded >= POSITIVE_I64_LIMIT) return false;
            long code = (long) rounded;
            float recovered = (float) (((double) code) / power);
            if (Float.floatToRawIntBits(recovered) != rawWords[i]) return false;
            codes[i] = code;
        }
        return true;
    }

    private static double powerOfTen(int scale) {
        return Math.pow(10.0, scale);
    }

    private static ByteArrayOutputStream header(byte[] magic, int mode, int scale, int count) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(HEADER_SIZE + Math.max(0, count));
        out.write(magic, 0, magic.length);
        out.write(mode);
        out.write(scale);
        writeU32LE(out, count);
        return out;
    }

    private static Header parseHeader(byte[] frame, byte[] magic) {
        if (frame == null || frame.length < HEADER_SIZE) {
            throw new IllegalArgumentException("DBF truncated header");
        }
        if (!hasMagic(frame, magic)) {
            throw new IllegalArgumentException("DBF invalid magic");
        }
        int mode = frame[4] & 0xff;
        int scale = frame[5] & 0xff;
        if (mode == MODE_SCALED) {
            if (scale > MAX_SCALE) throw new IllegalArgumentException("DBF1 invalid scaled-mode scale");
        } else if (mode == MODE_IEEE_BITS) {
            if (scale != RAW_SCALE) throw new IllegalArgumentException("DBF1 invalid raw-mode scale");
        } else {
            throw new IllegalArgumentException("DBF1 invalid mode");
        }
        long unsignedCount = readU32LE(frame, 6);
        if (unsignedCount > MAX_BLOCK_VALUES) {
            throw new IllegalArgumentException("DBF1 impossible value count");
        }
        int count = (int) unsignedCount;
        if (count == 0 && (mode != MODE_IEEE_BITS || scale != RAW_SCALE)) {
            throw new IllegalArgumentException("DBF1 noncanonical empty frame");
        }
        return new Header(mode, scale, count);
    }

    private static Header parseAnyHeader(byte[] frame) {
        if (hasMagic(frame, MAGIC64)) return parseHeader(frame, MAGIC64);
        if (hasMagic(frame, MAGIC32)) return parseHeader(frame, MAGIC32);
        throw new IllegalArgumentException("DBF invalid magic");
    }

    private static boolean hasMagic(byte[] frame, byte[] magic) {
        if (frame == null || frame.length < magic.length) return false;
        for (int i = 0; i < magic.length; i++) if (frame[i] != magic[i]) return false;
        return true;
    }

    private static byte[] encodeCodes(byte[] magic, int mode, int scale, long[] codes) {
        ByteArrayOutputStream out = header(magic, mode, scale, codes.length);
        writeLongLE(out, codes[0]);
        long previous = codes[0];
        for (int i = 1; i < codes.length; i++) {
            long current = codes[i];
            writeUnsignedLeb128(out, zigzag(current - previous));
            previous = current;
        }
        return out.toByteArray();
    }

    private static DecodedCodes decodeCodes(byte[] frame, byte[] magic) {
        Header header = parseHeader(frame, magic);
        int position = HEADER_SIZE;
        if (header.count == 0) {
            if (position != frame.length) throw new IllegalArgumentException("DBF trailing bytes");
            return new DecodedCodes(header, new long[0]);
        }
        long minimumBody = 8L + header.count - 1L;
        if ((long) frame.length - HEADER_SIZE < minimumBody) {
            throw new IllegalArgumentException("DBF truncated body");
        }
        long[] codes = new long[header.count];
        codes[0] = readLongLE(frame, position);
        Position cursor = new Position(position + Long.BYTES);
        long current = codes[0];
        for (int i = 1; i < codes.length; i++) {
            current += unzigzag(readCanonicalUnsignedLeb128(frame, cursor));
            codes[i] = current;
        }
        if (cursor.value != frame.length) throw new IllegalArgumentException("DBF trailing bytes");
        return new DecodedCodes(header, codes);
    }

    private static void writeU32LE(ByteArrayOutputStream out, int value) {
        for (int i = 0; i < Integer.BYTES; i++) out.write(value >>> (8 * i));
    }

    private static long readU32LE(byte[] input, int position) {
        long value = 0;
        for (int i = 0; i < Integer.BYTES; i++) value |= (long) (input[position + i] & 0xff) << (8 * i);
        return value;
    }

    private static void writeLongLE(ByteArrayOutputStream out, long value) {
        for (int i = 0; i < Long.BYTES; i++) out.write((int) (value >>> (8 * i)));
    }

    private static long readLongLE(byte[] input, int position) {
        if (input.length - position < Long.BYTES) throw new IllegalArgumentException("DBF1 truncated first value");
        long value = 0;
        for (int i = 0; i < Long.BYTES; i++) value |= (long) (input[position + i] & 0xff) << (8 * i);
        return value;
    }

    private static long zigzag(long value) {
        return (value << 1) ^ (value >> 63);
    }

    private static long unzigzag(long value) {
        return (value >>> 1) ^ -(value & 1L);
    }

    private static void writeUnsignedLeb128(ByteArrayOutputStream out, long value) {
        long remaining = value;
        while ((remaining & ~0x7fL) != 0) {
            out.write((int) ((remaining & 0x7fL) | 0x80L));
            remaining >>>= 7;
        }
        out.write((int) remaining);
    }

    private static long readCanonicalUnsignedLeb128(byte[] input, Position position) {
        long value = 0;
        for (int index = 0; index < 10; index++) {
            if (position.value >= input.length) throw new IllegalArgumentException("DBF1 truncated varint");
            int current = input[position.value++] & 0xff;
            int payload = current & 0x7f;
            if (index == 9 && payload > 1) throw new IllegalArgumentException("DBF1 overflowing varint");
            value |= (long) payload << (index * 7);
            if ((current & 0x80) == 0) {
                if (index > 0 && payload == 0) throw new IllegalArgumentException("DBF1 noncanonical varint");
                return value;
            }
        }
        throw new IllegalArgumentException("DBF1 overflowing varint");
    }

    private static final class Header {
        final int mode;
        final int scale;
        final int count;

        Header(int mode, int scale, int count) {
            this.mode = mode;
            this.scale = scale;
            this.count = count;
        }
    }

    private static final class DecodedCodes {
        final Header header;
        final long[] codes;

        DecodedCodes(Header header, long[] codes) {
            this.header = header;
            this.codes = codes;
        }
    }

    private static final class Position {
        int value;

        Position(int value) {
            this.value = value;
        }
    }
}
