package org.example;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

/** Lossless Java port of {@code cpp/src/pongo_codec.cpp}. */
public final class PongoDouble {
    public static final String ROW_NAME = "Pongo";
    private static final int MANTISSA_BITS = 52;
    private static final int EXPONENT_BIAS = 1023;
    private static final int MAX_DECIMAL_PLACES = 16;

    private PongoDouble() {}

    private static final class BitWriter {
        final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        int used;
        int current;

        void writeBit(boolean value) {
            if (value) current |= 1 << (7 - used);
            if (++used == 8) {
                bytes.write(current);
                used = 0;
                current = 0;
            }
        }

        void writeBits(long value, int count) {
            for (int bit = count - 1; bit >= 0; bit--) writeBit(((value >>> bit) & 1L) != 0);
        }

        byte[] finish() {
            if (used != 0) bytes.write(current);
            return bytes.toByteArray();
        }
    }

    private static final class BitReader {
        final byte[] bytes;
        int byteIndex;
        int used;
        boolean ok = true;

        BitReader(byte[] bytes, int offset) {
            this.bytes = bytes;
            this.byteIndex = offset;
        }

        boolean readBit() {
            if (!ok || byteIndex >= bytes.length) {
                ok = false;
                return false;
            }
            boolean value = (((bytes[byteIndex] & 0xff) >>> (7 - used)) & 1) != 0;
            if (++used == 8) {
                used = 0;
                byteIndex++;
            }
            return value;
        }

        long readBits(int count) {
            long value = 0;
            for (int i = 0; i < count; i++) {
                value = (value << 1) | (readBit() ? 1L : 0L);
                if (!ok) return 0;
            }
            return value;
        }
    }

    private static final class Reencoded {
        long bits;
        int leadingDecimalZeros;
    }

    private static long lowMask(int count) {
        return count <= 0 ? 0 : count >= 64 ? -1L : (1L << count) - 1;
    }

    private static long reverseLowBits(long value, int count) {
        long reversed = 0;
        for (int i = 0; i < count; i++) reversed = (reversed << 1) | ((value >>> i) & 1L);
        return reversed;
    }

    private static String[] decimalParts(double value) {
        String text = Double.toString(Math.abs(value));
        int epos = Math.max(text.indexOf('e'), text.indexOf('E'));
        String mantissa = epos < 0 ? text : text.substring(0, epos);
        int exponent = epos < 0 ? 0 : Integer.parseInt(text.substring(epos + 1));
        int dot = mantissa.indexOf('.');
        String digits = dot < 0 ? mantissa : mantissa.substring(0, dot) + mantissa.substring(dot + 1);
        int decimalIndex = (dot < 0 ? digits.length() : dot) + exponent;
        String integer;
        String fraction;
        if (decimalIndex <= 0) {
            integer = "0";
            fraction = "0".repeat(-decimalIndex) + digits;
        } else if (decimalIndex >= digits.length()) {
            integer = digits + "0".repeat(decimalIndex - digits.length());
            fraction = "";
        } else {
            integer = digits.substring(0, decimalIndex);
            fraction = digits.substring(decimalIndex);
        }
        integer = integer.replaceFirst("^0+(?!$)", "");
        fraction = fraction.replaceFirst("0+$", "");
        return new String[] {integer, fraction};
    }

    private static boolean restore(long reencodedBits, int leadingZeros, long[] result) {
        int biased = (int) ((reencodedBits >>> 52) & 0x7ffL);
        if (biased == 0x7ff) return false;
        int available = MANTISSA_BITS - Math.max(0, biased - EXPONENT_BIAS);
        if (available <= 0) return false;
        long field = reencodedBits & lowMask(available);
        if (field == 0) return false;
        int trailing = Long.numberOfTrailingZeros(field);
        int fractionBits = available - trailing;
        long decimalFraction = reverseLowBits(field >>> trailing, fractionBits);
        if (decimalFraction == 0) return false;
        double reencoded = Double.longBitsToDouble(reencodedBits);
        String decimal = (reencodedBits < 0 ? "-" : "")
                + Long.toString((long) Math.floor(Math.abs(reencoded))) + "."
                + "0".repeat(leadingZeros) + Long.toString(decimalFraction);
        try {
            result[0] = Double.doubleToRawLongBits(Double.parseDouble(decimal));
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static boolean tryReencode(double value, Reencoded out) {
        if (!Double.isFinite(value) || value == 0.0) return false;
        String fraction = decimalParts(value)[1];
        if (fraction.isEmpty() || fraction.length() > MAX_DECIMAL_PLACES) return false;
        int first = 0;
        while (first < fraction.length() && fraction.charAt(first) == '0') first++;
        if (first == fraction.length() || first > 15) return false;
        long decimalFraction;
        try {
            decimalFraction = Long.parseLong(fraction.substring(first));
        } catch (NumberFormatException e) {
            return false;
        }
        long original = Double.doubleToRawLongBits(value);
        int biased = (int) ((original >>> 52) & 0x7ffL);
        int available = MANTISSA_BITS - Math.max(0, biased - EXPONENT_BIAS);
        int fractionBits = 64 - Long.numberOfLeadingZeros(decimalFraction);
        if (available <= 0 || fractionBits > available) return false;
        long field = reverseLowBits(decimalFraction, fractionBits) << (available - fractionBits);
        long reencoded = (original & ~lowMask(available)) | field;
        long[] restored = new long[1];
        if (!restore(reencoded, first, restored) || restored[0] != original) return false;
        out.bits = reencoded;
        out.leadingDecimalZeros = first;
        return true;
    }

    private static void writeXor(BitWriter writer, long xor, int[] window) {
        if (xor == 0) {
            writer.writeBit(false);
            return;
        }
        writer.writeBit(true);
        int leading = Long.numberOfLeadingZeros(xor);
        int trailing = Long.numberOfTrailingZeros(xor);
        if (window[0] >= 0 && leading >= window[0] && trailing >= window[1]) {
            writer.writeBit(false);
            writer.writeBits(xor >>> window[1], 64 - window[0] - window[1]);
            return;
        }
        writer.writeBit(true);
        int significant = 64 - leading - trailing;
        writer.writeBits(leading, 6);
        writer.writeBits(significant == 64 ? 0 : significant, 6);
        writer.writeBits(xor >>> trailing, significant);
        window[0] = leading;
        window[1] = trailing;
    }

    public static byte[] encode(double[] values) {
        BitWriter writer = new BitWriter();
        for (int i = 0; i < 8; i++) writer.writeBits(((long) values.length >>> (8 * i)) & 0xffL, 8);
        if (values.length == 0) return writer.finish();
        long stored = Double.doubleToRawLongBits(values[0]);
        writer.writeBits(stored, 64);
        int[] window = {-1, 0};
        for (int i = 1; i < values.length; i++) {
            long original = Double.doubleToRawLongBits(values[i]);
            Reencoded re = new Reencoded();
            if (tryReencode(values[i], re)) {
                if (re.leadingDecimalZeros == 0) writer.writeBit(true);
                else {
                    writer.writeBits(0, 2);
                    writer.writeBits(re.leadingDecimalZeros, 4);
                }
                writeXor(writer, stored ^ re.bits, window);
                stored = re.bits;
            } else {
                writer.writeBits(1, 2);
                writeXor(writer, stored ^ original, window);
                stored = original;
            }
        }
        return writer.finish();
    }

    public static double[] decode(byte[] encoded) {
        if (encoded.length < 8) return null;
        long countLong = 0;
        for (int i = 0; i < 8; i++) countLong |= (long) (encoded[i] & 0xff) << (8 * i);
        if (countLong < 0 || countLong > encoded.length * 8L + 1 || countLong > Integer.MAX_VALUE) return null;
        int count = (int) countLong;
        if (count == 0) return encoded.length == 8 ? new double[0] : null;
        BitReader reader = new BitReader(encoded, 8);
        long stored = reader.readBits(64);
        if (!reader.ok) return null;
        double[] out = new double[count];
        out[0] = Double.longBitsToDouble(stored);
        int leading = -1, trailing = 0;
        for (int i = 1; i < count; i++) {
            boolean reencoded;
            int decimalZeros = 0;
            if (reader.readBit()) reencoded = true;
            else if (reader.readBit()) reencoded = false;
            else {
                reencoded = true;
                decimalZeros = (int) reader.readBits(4);
            }
            if (!reader.ok || !reader.readBit()) {
                if (!reader.ok) return null;
            } else {
                boolean newWindow = reader.readBit();
                if (!reader.ok) return null;
                if (newWindow) {
                    leading = (int) reader.readBits(6);
                    int significant = (int) reader.readBits(6);
                    if (significant == 0) significant = 64;
                    trailing = 64 - leading - significant;
                    if (!reader.ok || trailing < 0) return null;
                } else if (leading < 0) return null;
                int significant = 64 - leading - trailing;
                stored ^= reader.readBits(significant) << trailing;
                if (!reader.ok) return null;
            }
            long original = stored;
            if (reencoded) {
                long[] restored = new long[1];
                if (!restore(stored, decimalZeros, restored)) return null;
                original = restored[0];
            }
            out[i] = Double.longBitsToDouble(original);
        }
        return reader.ok ? out : null;
    }

    public static void bench(List<? extends List<Double>> columns, long[] out, double[] lossAcc) {
        long original = 0, compressed = 0, compressNanos = 0, decompressNanos = 0;
        try {
            for (List<Double> column : columns) {
                if (column.isEmpty()) continue;
                double[] values = new double[column.size()];
                for (int i = 0; i < values.length; i++) values[i] = column.get(i) == null ? Double.NaN : column.get(i);
                long t0 = System.nanoTime();
                byte[] bytes = encode(values);
                long t1 = System.nanoTime();
                double[] decoded = decode(bytes);
                long t2 = System.nanoTime();
                if (decoded == null || decoded.length != values.length) continue;
                List<Double> decodedList = lossAcc == null ? null : new ArrayList<>(decoded.length);
                for (int i = 0; i < values.length; i++) {
                    if (Double.doubleToRawLongBits(values[i]) != Double.doubleToRawLongBits(decoded[i])) throw new IllegalStateException("Pongo round trip mismatch");
                    if (decodedList != null) decodedList.add(decoded[i]);
                }
                if (lossAcc != null) BenchmarkLoss.accumulateDoubleListLoss(column, decodedList, lossAcc);
                original += BenchmarkWireFormat.columnOriginalSize(values.length, true);
                compressed += bytes.length;
                compressNanos += t1 - t0;
                decompressNanos += t2 - t1;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = original;
        out[1] = compressed;
        out[2] = compressNanos;
        out[3] = decompressNanos;
    }
}
