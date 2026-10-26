package org.example;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/** Binary stdin/stdout bridge for independent DBF1 encode/decode and a small local benchmark. */
public final class DbpFloat64Cli {
    private DbpFloat64Cli() {}

    public static void main(String[] args) throws IOException {
        if (args.length > 0 && "bench-manifest".equals(args[0])) {
            DbpFloatManifestBench.run(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        if (args.length < 1 || args.length > 2) {
            throw new IllegalArgumentException(
                    "usage: DbpFloat64Cli encode|encode32 MAX_POINT | decode | bench|bench32 MAX_POINT"
                            + " | bench-manifest --manifest TSV --aggregates JSONL --output JSONL --repeats 1");
        }
        String command = args[0];
        if ("decode".equals(command)) {
            if (args.length != 1) throw new IllegalArgumentException("decode takes no max point");
            byte[] input = readBounded(System.in, maximumFrameBytes());
            if (DbpFloat64Codec.isFloat32Frame(input)) {
                System.out.write(toLittleEndianFloatWords(DbpFloat64Codec.decodeFrame32(input)));
            } else {
                System.out.write(toLittleEndianDoubles(DbpFloat64Codec.decodeFrame(input)));
            }
            return;
        }
        if (args.length != 2) throw new IllegalArgumentException(command + " requires max point");
        int maxPoint = Integer.parseInt(args[1]);
        if ("encode".equals(command)) {
            byte[] input = readBounded(System.in, DbpFloat64Codec.MAX_BLOCK_VALUES * Double.BYTES);
            double[] values = fromLittleEndianDoubles(input);
            System.out.write(DbpFloat64Codec.encodeFrame(values, maxPoint));
            return;
        }
        if ("encode32".equals(command)) {
            byte[] input = readBounded(System.in, DbpFloat64Codec.MAX_BLOCK_VALUES * Float.BYTES);
            System.out.write(DbpFloat64Codec.encodeFrame32(fromLittleEndianFloatWords(input), maxPoint));
            return;
        }
        if ("bench".equals(command)) {
            runBenchmark(System.in, maxPoint);
            return;
        }
        if ("bench32".equals(command)) {
            runBenchmark32(System.in, maxPoint);
            return;
        }
        throw new IllegalArgumentException("unknown command: " + command);
    }

    private static void runBenchmark32(InputStream input, int maxPoint) throws IOException {
        final int rawBlockBytes = DbpFloat64Codec.MAX_BLOCK_VALUES * Float.BYTES;
        long validCount = 0;
        long compressed = 0, encodeNanos = 0, decodeNanos = 0;
        int scaledBlocks = 0, rawBlocks = 0;
        boolean warmed = false;
        while (true) {
            byte[] raw = readBlock(input, rawBlockBytes);
            if (raw.length == 0) break;
            int[] block = fromLittleEndianFloatWords(raw);
            if (!warmed) {
                byte[] warmFrame = DbpFloat64Codec.encodeFrame32(block, maxPoint);
                requireRawEqual(block, DbpFloat64Codec.decodeFrame32(warmFrame));
                warmed = true;
            }
            long t0 = System.nanoTime();
            byte[] frame = DbpFloat64Codec.encodeFrame32(block, maxPoint);
            long t1 = System.nanoTime();
            int[] decoded = DbpFloat64Codec.decodeFrame32(frame);
            long t2 = System.nanoTime();
            requireRawEqual(block, decoded);
            compressed += frame.length;
            encodeNanos += t1 - t0;
            decodeNanos += t2 - t1;
            validCount += block.length;
            if (DbpFloat64Codec.modeOf(frame) == DbpFloat64Codec.MODE_SCALED) scaledBlocks++;
            else rawBlocks++;
        }
        if (!warmed) {
            int[] words = new int[0];
            byte[] warmFrame = DbpFloat64Codec.encodeFrame32(words, maxPoint);
            requireRawEqual(words, DbpFloat64Codec.decodeFrame32(warmFrame));
            long t0 = System.nanoTime();
            byte[] frame = DbpFloat64Codec.encodeFrame32(words, maxPoint);
            long t1 = System.nanoTime();
            int[] decoded = DbpFloat64Codec.decodeFrame32(frame);
            long t2 = System.nanoTime();
            requireRawEqual(words, decoded);
            compressed = frame.length;
            encodeNanos = t1 - t0;
            decodeNanos = t2 - t1;
            rawBlocks = 1;
        }
        printBenchmarkJson(
                validCount,
                Float.BYTES,
                compressed,
                encodeNanos,
                decodeNanos,
                scaledBlocks,
                rawBlocks);
    }

    private static void runBenchmark(InputStream input, int maxPoint) throws IOException {
        final int rawBlockBytes = DbpFloat64Codec.MAX_BLOCK_VALUES * Double.BYTES;
        long validCount = 0;
        long compressed = 0, encodeNanos = 0, decodeNanos = 0;
        int scaledBlocks = 0, rawBlocks = 0;
        boolean warmed = false;
        while (true) {
            byte[] raw = readBlock(input, rawBlockBytes);
            if (raw.length == 0) break;
            double[] block = fromLittleEndianDoubles(raw);
            if (!warmed) {
                byte[] warmFrame = DbpFloat64Codec.encodeFrame(block, maxPoint);
                requireRawEqual(block, DbpFloat64Codec.decodeFrame(warmFrame));
                warmed = true;
            }
            long t0 = System.nanoTime();
            byte[] frame = DbpFloat64Codec.encodeFrame(block, maxPoint);
            long t1 = System.nanoTime();
            double[] decoded = DbpFloat64Codec.decodeFrame(frame);
            long t2 = System.nanoTime();
            requireRawEqual(block, decoded);
            compressed += frame.length;
            encodeNanos += t1 - t0;
            decodeNanos += t2 - t1;
            validCount += block.length;
            if (DbpFloat64Codec.modeOf(frame) == DbpFloat64Codec.MODE_SCALED) scaledBlocks++;
            else rawBlocks++;
        }
        if (!warmed) {
            double[] values = new double[0];
            byte[] warmFrame = DbpFloat64Codec.encodeFrame(values, maxPoint);
            requireRawEqual(values, DbpFloat64Codec.decodeFrame(warmFrame));
            long t0 = System.nanoTime();
            byte[] frame = DbpFloat64Codec.encodeFrame(values, maxPoint);
            long t1 = System.nanoTime();
            double[] decoded = DbpFloat64Codec.decodeFrame(frame);
            long t2 = System.nanoTime();
            requireRawEqual(values, decoded);
            compressed = frame.length;
            encodeNanos = t1 - t0;
            decodeNanos = t2 - t1;
            rawBlocks = 1;
        }
        printBenchmarkJson(
                validCount,
                Double.BYTES,
                compressed,
                encodeNanos,
                decodeNanos,
                scaledBlocks,
                rawBlocks);
    }

    private static void printBenchmarkJson(
            long validCount,
            int widthBytes,
            long compressed,
            long encodeNanos,
            long decodeNanos,
            int scaledBlocks,
            int rawBlocks) {
        System.out.printf(
                "{\"valid_count\":%d,\"original_bytes\":%d,\"compressed_bytes\":%d,"
                        + "\"encode_ns\":%d,\"decode_ns\":%d,\"scaled_blocks\":%d,"
                        + "\"raw_blocks\":%d,\"verified_values\":%d}%n",
                validCount,
                validCount * widthBytes,
                compressed,
                encodeNanos,
                decodeNanos,
                scaledBlocks,
                rawBlocks,
                validCount);
    }

    private static byte[] readBlock(InputStream input, int maximumBytes) throws IOException {
        byte[] buffer = new byte[maximumBytes];
        int size = 0;
        while (size < buffer.length) {
            int read = input.read(buffer, size, buffer.length - size);
            if (read < 0) break;
            if (read == 0) continue;
            size += read;
        }
        return size == buffer.length ? buffer : Arrays.copyOf(buffer, size);
    }

    private static byte[] readBounded(InputStream input, int maximumBytes) throws IOException {
        byte[] result = input.readNBytes(maximumBytes + 1);
        if (result.length > maximumBytes) throw new IllegalArgumentException("input exceeds one DBF1 block");
        return result;
    }

    private static int maximumFrameBytes() {
        return DbpFloat64Codec.HEADER_SIZE + Long.BYTES
                + (DbpFloat64Codec.MAX_BLOCK_VALUES - 1) * 10;
    }

    private static double[] fromLittleEndianDoubles(byte[] raw) {
        if (raw.length % Double.BYTES != 0) {
            throw new IllegalArgumentException("float64 input has a partial lane");
        }
        double[] values = new double[raw.length / Double.BYTES];
        for (int i = 0; i < values.length; i++) {
            long bits = 0;
            for (int b = 0; b < Long.BYTES; b++) bits |= (long) (raw[i * 8 + b] & 0xff) << (8 * b);
            values[i] = Double.longBitsToDouble(bits);
        }
        return values;
    }

    private static int[] fromLittleEndianFloatWords(byte[] raw) {
        if (raw.length % Float.BYTES != 0) {
            throw new IllegalArgumentException("float32 input has a partial lane");
        }
        int[] words = new int[raw.length / Float.BYTES];
        for (int i = 0; i < words.length; i++) {
            int word = 0;
            for (int b = 0; b < Integer.BYTES; b++) word |= (raw[i * 4 + b] & 0xff) << (8 * b);
            words[i] = word;
        }
        return words;
    }

    private static byte[] toLittleEndianDoubles(double[] values) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(values.length * Double.BYTES);
        for (double value : values) {
            long bits = Double.doubleToRawLongBits(value);
            for (int b = 0; b < Long.BYTES; b++) out.write((int) (bits >>> (8 * b)));
        }
        return out.toByteArray();
    }

    private static byte[] toLittleEndianFloatWords(int[] words) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(words.length * Float.BYTES);
        for (int word : words) {
            for (int b = 0; b < Integer.BYTES; b++) out.write(word >>> (8 * b));
        }
        return out.toByteArray();
    }

    private static void requireRawEqual(double[] expected, double[] actual) {
        if (expected.length != actual.length) throw new IllegalStateException("DBF1 count mismatch");
        for (int i = 0; i < expected.length; i++) {
            if (Double.doubleToRawLongBits(expected[i]) != Double.doubleToRawLongBits(actual[i])) {
                throw new IllegalStateException("DBF1 bitwise mismatch at " + i);
            }
        }
    }

    private static void requireRawEqual(int[] expected, int[] actual) {
        if (!Arrays.equals(expected, actual)) throw new IllegalStateException("DBF2 bitwise mismatch");
    }
}
