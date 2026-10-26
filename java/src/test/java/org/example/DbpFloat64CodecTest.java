package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Standalone DBF1 golden, boundary, and malformed-frame regression. */
public final class DbpFloat64CodecTest {
    private DbpFloat64CodecTest() {}

    public static void main(String[] args) {
        checkGolden(
                new double[0],
                18,
                "4442463101ff00000000",
                DbpFloat64Codec.MODE_IEEE_BITS,
                DbpFloat64Codec.RAW_SCALE);
        checkGolden(
                new double[] {0.0, 1.25, 1.5},
                2,
                "444246310002030000000000000000000000fa0132",
                DbpFloat64Codec.MODE_SCALED,
                2);
        checkGolden(
                bits(0x8000000000000000L, 0x7ff0000000000000L, 0x7ff8000000000042L),
                18,
                "4442463101ff030000000000000000000080ffffffffffffff0f8481808080808008",
                DbpFloat64Codec.MODE_IEEE_BITS,
                DbpFloat64Codec.RAW_SCALE);
        checkGolden(
                bits(0x8000000000000000L, 0x7fffffffffffffffL),
                18,
                "4442463101ff02000000000000000000008001",
                DbpFloat64Codec.MODE_IEEE_BITS,
                DbpFloat64Codec.RAW_SCALE);
        checkGolden32(
                new int[] {0x00000000, 0x3fa00000, 0x3fc00000},
                2,
                "444246320002030000000000000000000000fa0132",
                DbpFloat64Codec.MODE_SCALED,
                2);
        checkGolden32(
                new int[] {0x80000000, 0x7f800000, 0x7fc00042},
                18,
                "4442463201ff030000000000008000000000ffffff0784818004",
                DbpFloat64Codec.MODE_IEEE_BITS,
                DbpFloat64Codec.RAW_SCALE);

        double[] rawCases = bits(
                0x0000000000000000L,
                0x8000000000000000L,
                0x0000000000000001L,
                0x0010000000000000L,
                0x7fefffffffffffffL,
                0xffefffffffffffffL,
                0x7ff0000000000000L,
                0xfff0000000000000L,
                0x7ff8000000000042L,
                0xfff8000000001234L);
        byte[] rawFrame = DbpFloat64Codec.encodeFrame(rawCases, -1);
        check(DbpFloat64Codec.modeOf(rawFrame) == DbpFloat64Codec.MODE_IEEE_BITS, "raw mode");
        checkRawEqual(rawCases, DbpFloat64Codec.decodeFrame(rawFrame));

        int[] rawCases32 = {
            0x00000000,
            0x80000000,
            0x00000001,
            0x00800000,
            0x7f7fffff,
            0xff7fffff,
            0x7f800000,
            0xff800000,
            0x7fc00042,
            0xffc01234
        };
        byte[] rawFrame32 = DbpFloat64Codec.encodeFrame32(rawCases32, -1);
        check(DbpFloat64Codec.modeOf(rawFrame32) == DbpFloat64Codec.MODE_IEEE_BITS, "float32 raw mode");
        check(Arrays.equals(rawCases32, DbpFloat64Codec.decodeFrame32(rawFrame32)), "float32 specials");
        ArrayList<Float> list32 = new ArrayList<>();
        for (int word : rawCases32) list32.add(Float.intBitsToFloat(word));
        long[] total32 = new long[4];
        double[] loss32 = new double[2];
        ColumnarEncodings.benchDbpFloat32(List.of(list32), List.of(-1), total32, loss32);
        check(total32[0] == (long) rawCases32.length * Float.BYTES, "float32 benchmark original size");
        check(total32[1] == rawFrame32.length, "float32 benchmark framed size");
        check(loss32[0] == 0.0 && loss32[1] == rawCases32.length, "float32 benchmark bitwise loss");

        ArrayList<Double> list = new ArrayList<>();
        for (double value : rawCases) list.add(value);
        long[] total = new long[4];
        double[] loss = new double[2];
        ColumnarEncodings.benchDbpFloat64(List.of(list), List.of(-1), total, loss);
        check(total[0] == (long) rawCases.length * Double.BYTES, "benchmark original size");
        check(total[1] == rawFrame.length, "benchmark framed size");
        check(loss[0] == 0.0 && loss[1] == rawCases.length, "benchmark bitwise loss");

        byte[] scaled = DbpFloat64Codec.encodeFrame(new double[] {0.0, 1.25, 1.5}, 2);
        expectBad(Arrays.copyOf(scaled, scaled.length - 1), "truncation");
        byte[] trailing = Arrays.copyOf(scaled, scaled.length + 1);
        trailing[trailing.length - 1] = 0;
        expectBad(trailing, "trailing byte");
        byte[] badMagic = scaled.clone();
        badMagic[0] = 'X';
        expectBad(badMagic, "magic");
        byte[] badMode = scaled.clone();
        badMode[4] = 2;
        expectBad(badMode, "mode");
        byte[] badScale = scaled.clone();
        badScale[5] = 19;
        expectBad(badScale, "scale");
        expectBad(fromHex("4442463100000200000000000000000000008000"), "noncanonical varint");
        expectBad(fromHex("444246310000020000000000000000000000ffffffffffffffffff02"), "overflow varint");
        expectBad(fromHex("4442463101ff11270000"), "count above block limit");
        expectBad(fromHex("44424631000000000000"), "noncanonical empty frame");
        expectBad32(fromHex("4442463201ff010000000000000001000000"), "raw word above uint32");

        expectThrows(() -> DbpFloat64Codec.encodeFrame(new double[10_001], 0), "oversized block");
        expectThrows(() -> DbpFloat64Codec.encodeFrame32(new int[10_001], 0), "oversized float32 block");
        System.out.println("dbp_float_java_goldens=6 status=PASS");
    }

    private static void checkGolden32(int[] words, int maxPoint, String expectedHex, int mode, int scale) {
        byte[] encoded = DbpFloat64Codec.encodeFrame32(words, maxPoint);
        check(toHex(encoded).equals(expectedHex), "float32 golden payload: " + toHex(encoded));
        check(DbpFloat64Codec.isFloat32Frame(encoded), "float32 magic");
        check(DbpFloat64Codec.modeOf(encoded) == mode, "float32 golden mode");
        check(DbpFloat64Codec.scaleOf(encoded) == scale, "float32 golden scale");
        check(DbpFloat64Codec.countOf(encoded) == words.length, "float32 golden count");
        check(Arrays.equals(words, DbpFloat64Codec.decodeFrame32(encoded)), "float32 raw mismatch");
    }

    private static void checkGolden(double[] values, int maxPoint, String expectedHex, int mode, int scale) {
        byte[] encoded = DbpFloat64Codec.encodeFrame(values, maxPoint);
        check(toHex(encoded).equals(expectedHex), "golden payload: " + toHex(encoded));
        check(DbpFloat64Codec.modeOf(encoded) == mode, "golden mode");
        check(DbpFloat64Codec.scaleOf(encoded) == scale, "golden scale");
        check(DbpFloat64Codec.countOf(encoded) == values.length, "golden count");
        checkRawEqual(values, DbpFloat64Codec.decodeFrame(encoded));
    }

    private static double[] bits(long... words) {
        double[] values = new double[words.length];
        for (int i = 0; i < words.length; i++) values[i] = Double.longBitsToDouble(words[i]);
        return values;
    }

    private static void checkRawEqual(double[] expected, double[] actual) {
        check(expected.length == actual.length, "decoded count");
        for (int i = 0; i < expected.length; i++) {
            check(
                    Double.doubleToRawLongBits(expected[i]) == Double.doubleToRawLongBits(actual[i]),
                    "raw mismatch at " + i);
        }
    }

    private static void expectBad(byte[] frame, String label) {
        expectThrows(() -> DbpFloat64Codec.decodeFrame(frame), label);
    }

    private static void expectBad32(byte[] frame, String label) {
        expectThrows(() -> DbpFloat64Codec.decodeFrame32(frame), label);
    }

    private static void expectThrows(Runnable action, String label) {
        try {
            action.run();
            throw new AssertionError("expected rejection: " + label);
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static String toHex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) out.append(String.format("%02x", value & 0xff));
        return out.toString();
    }

    private static byte[] fromHex(String hex) {
        if ((hex.length() & 1) != 0) throw new IllegalArgumentException("odd hex");
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }
}
