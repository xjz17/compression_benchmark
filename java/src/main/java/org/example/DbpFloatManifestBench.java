package org.example;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.Set;

/** Streaming DBF1/DBF2 benchmark over the frozen per-dataset block manifest. */
final class DbpFloatManifestBench {
    private static final int TSV_FIELDS = 15;
    private static final int WARMUP_BLOCKS = 20;

    private DbpFloatManifestBench() {}

    static void run(String[] args) throws IOException {
        Arguments parsed = Arguments.parse(args);
        if (!Files.isRegularFile(parsed.manifest)) {
            throw new IllegalArgumentException("DBF manifest is not a regular file: " + parsed.manifest);
        }
        if (!Files.isRegularFile(parsed.aggregates)) {
            throw new IllegalArgumentException("DBF aggregate inventory is not a regular file: " + parsed.aggregates);
        }
        warmFirstBlocks(parsed.manifest);
        benchmark(parsed.manifest, parsed.output);
    }

    private static void warmFirstBlocks(Path manifest) throws IOException {
        Path base = manifest.toAbsolutePath().normalize().getParent();
        try (BufferedReader reader = Files.newBufferedReader(manifest, StandardCharsets.UTF_8);
                SegmentReader valueSegments = new SegmentReader(base);
                SegmentReader missingSegments = new SegmentReader(base)) {
            String line;
            int lineNumber = 0;
            int warmed = 0;
            while (warmed < WARMUP_BLOCKS && (line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isEmpty()) throw new IllegalArgumentException("empty DBF TSV row at " + lineNumber);
                BlockRow row = BlockRow.parse(line, lineNumber);
                runBlock(row, valueSegments, missingSegments, false);
                warmed++;
            }
        }
    }

    private static void benchmark(Path manifest, Path output) throws IOException {
        Path absoluteOutput = output.toAbsolutePath().normalize();
        Path outputParent = absoluteOutput.getParent();
        if (outputParent != null) Files.createDirectories(outputParent);
        Path temporary = absoluteOutput.resolveSibling(absoluteOutput.getFileName() + ".tmp");
        Path base = manifest.toAbsolutePath().normalize().getParent();
        boolean complete = false;
        try (BufferedReader reader = Files.newBufferedReader(manifest, StandardCharsets.UTF_8);
                BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8);
                SegmentReader valueSegments = new SegmentReader(base);
                SegmentReader missingSegments = new SegmentReader(base)) {
            Set<String> finished = new HashSet<>();
            Aggregate aggregate = null;
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isEmpty()) throw new IllegalArgumentException("empty DBF TSV row at " + lineNumber);
                BlockRow row = BlockRow.parse(line, lineNumber);
                if (aggregate == null || !aggregate.id.equals(row.aggregateId)) {
                    if (aggregate != null) {
                        writer.write(aggregate.toJson());
                        writer.newLine();
                        finished.add(aggregate.id);
                    }
                    if (finished.contains(row.aggregateId)) {
                        throw new IllegalArgumentException(
                                "noncontiguous DBF aggregate at TSV line " + lineNumber + ": " + row.aggregateId);
                    }
                    aggregate = new Aggregate(row.aggregateId, row.width);
                }
                if (aggregate.width != row.width) {
                    throw new IllegalArgumentException("mixed widths for aggregate " + row.aggregateId);
                }
                aggregate.add(row, runBlock(row, valueSegments, missingSegments, true));
            }
            if (aggregate != null) {
                writer.write(aggregate.toJson());
                writer.newLine();
            }
            complete = true;
        } finally {
            if (!complete) Files.deleteIfExists(temporary);
        }
        try {
            Files.move(
                    temporary,
                    absoluteOutput,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(temporary, absoluteOutput, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static BlockResult runBlock(
            BlockRow row,
            SegmentReader valueSegments,
            SegmentReader missingSegments,
            boolean measured)
            throws IOException {
        byte[] raw = valueSegments.read(row.valuesPath, row.valuesOffset, row.valuesLength);
        requireSha256(raw, row.valuesSha256, "values", row.aggregateId);
        byte[] missing = missingSegments.read(row.missingPath, row.missingOffset, row.missingLength);
        requireSha256(missing, row.missingSha256, "missing bitmap", row.aggregateId);
        if (row.width == Float.BYTES) {
            int[] words = floatWords(raw);
            long t0 = measured ? System.nanoTime() : 0;
            byte[] frame = DbpFloat64Codec.encodeFrame32(words, row.maxPoint);
            long t1 = measured ? System.nanoTime() : 0;
            int[] decoded = DbpFloat64Codec.decodeFrame32(frame);
            long t2 = measured ? System.nanoTime() : 0;
            if (!java.util.Arrays.equals(words, decoded)) {
                throw new IllegalStateException("DBF2 bitwise verification failed for " + row.aggregateId);
            }
            return new BlockResult(frame, words.length, t1 - t0, t2 - t1);
        }
        double[] values = doubles(raw);
        long t0 = measured ? System.nanoTime() : 0;
        byte[] frame = DbpFloat64Codec.encodeFrame(values, row.maxPoint);
        long t1 = measured ? System.nanoTime() : 0;
        double[] decoded = DbpFloat64Codec.decodeFrame(frame);
        long t2 = measured ? System.nanoTime() : 0;
        requireRawEqual(values, decoded, row.aggregateId);
        return new BlockResult(frame, values.length, t1 - t0, t2 - t1);
    }

    private static void requireSha256(byte[] bytes, String expected, String kind, String aggregateId) {
        final MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
        byte[] actual = digest.digest(bytes);
        if (!hex(actual).equalsIgnoreCase(expected)) {
            throw new IllegalArgumentException(kind + " SHA-256 mismatch for " + aggregateId);
        }
    }

    private static String hex(byte[] bytes) {
        char[] digits = "0123456789abcdef".toCharArray();
        char[] out = new char[bytes.length * 2];
        for (int i = 0; i < bytes.length; i++) {
            int value = bytes[i] & 0xff;
            out[i * 2] = digits[value >>> 4];
            out[i * 2 + 1] = digits[value & 15];
        }
        return new String(out);
    }

    private static MessageDigest newSha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static int[] floatWords(byte[] raw) {
        if (raw.length % Float.BYTES != 0) throw new IllegalArgumentException("partial float32 lane");
        int[] words = new int[raw.length / Float.BYTES];
        for (int i = 0; i < words.length; i++) {
            int word = 0;
            for (int b = 0; b < Integer.BYTES; b++) word |= (raw[i * 4 + b] & 0xff) << (8 * b);
            words[i] = word;
        }
        return words;
    }

    private static double[] doubles(byte[] raw) {
        if (raw.length % Double.BYTES != 0) throw new IllegalArgumentException("partial float64 lane");
        double[] values = new double[raw.length / Double.BYTES];
        for (int i = 0; i < values.length; i++) {
            long bits = 0;
            for (int b = 0; b < Long.BYTES; b++) bits |= (long) (raw[i * 8 + b] & 0xff) << (8 * b);
            values[i] = Double.longBitsToDouble(bits);
        }
        return values;
    }

    private static void requireRawEqual(double[] expected, double[] actual, String aggregateId) {
        if (expected.length != actual.length) {
            throw new IllegalStateException("DBF1 count verification failed for " + aggregateId);
        }
        for (int i = 0; i < expected.length; i++) {
            if (Double.doubleToRawLongBits(expected[i]) != Double.doubleToRawLongBits(actual[i])) {
                throw new IllegalStateException(
                        "DBF1 bitwise verification failed for " + aggregateId + " at value " + i);
            }
        }
    }

    private static final class SegmentReader implements AutoCloseable {
        private final Path base;
        private Path openPath;
        private RandomAccessFile openFile;

        SegmentReader(Path base) {
            this.base = base;
        }

        byte[] read(String textPath, long offset, int length) throws IOException {
            if (length == 0) return new byte[0];
            Path path = Path.of(textPath);
            if (!path.isAbsolute()) path = base.resolve(path);
            path = path.normalize();
            if (!path.equals(openPath)) {
                closeOpenFile();
                openFile = new RandomAccessFile(path.toFile(), "r");
                openPath = path;
            }
            if (offset > openFile.length() || (long) length > openFile.length() - offset) {
                throw new IllegalArgumentException("DBF values segment exceeds file: " + path);
            }
            byte[] bytes = new byte[length];
            openFile.seek(offset);
            openFile.readFully(bytes);
            return bytes;
        }

        @Override
        public void close() throws IOException {
            closeOpenFile();
        }

        private void closeOpenFile() throws IOException {
            if (openFile != null) openFile.close();
            openFile = null;
            openPath = null;
        }
    }

    private static final class BlockRow {
        final String aggregateId;
        final int width;
        final int maxPoint;
        final String valuesPath;
        final long valuesOffset;
        final int valuesLength;
        final String valuesSha256;
        final String missingPath;
        final long missingOffset;
        final String missingSha256;
        final long rowCount;
        final long validCount;
        final long missingCount;
        final int missingLength;

        BlockRow(
                String aggregateId,
                int width,
                int maxPoint,
                String valuesPath,
                long valuesOffset,
                int valuesLength,
                String valuesSha256,
                String missingPath,
                long missingOffset,
                String missingSha256,
                long rowCount,
                long validCount,
                long missingCount,
                int missingLength) {
            this.aggregateId = aggregateId;
            this.width = width;
            this.maxPoint = maxPoint;
            this.valuesPath = valuesPath;
            this.valuesOffset = valuesOffset;
            this.valuesLength = valuesLength;
            this.valuesSha256 = valuesSha256;
            this.missingPath = missingPath;
            this.missingOffset = missingOffset;
            this.missingSha256 = missingSha256;
            this.rowCount = rowCount;
            this.validCount = validCount;
            this.missingCount = missingCount;
            this.missingLength = missingLength;
        }

        static BlockRow parse(String line, int lineNumber) {
            String[] fields = line.split("\\t", -1);
            if (fields.length != TSV_FIELDS) {
                throw new IllegalArgumentException(
                        "DBF TSV line " + lineNumber + " has " + fields.length + " fields, expected 15");
            }
            String aggregateId = fields[0];
            if (aggregateId.isEmpty()) throw new IllegalArgumentException("empty aggregate_id at " + lineNumber);
            int width = parseInt(fields[1], "width", lineNumber);
            if (width != Float.BYTES && width != Double.BYTES) {
                throw new IllegalArgumentException("width must be 4 or 8 at TSV line " + lineNumber);
            }
            int maxPoint = parseInt(fields[2], "max_point", lineNumber);
            long valuesOffset = parseLong(fields[4], "values_offset", lineNumber);
            long valuesLengthLong = parseLong(fields[5], "values_length", lineNumber);
            long missingOffset = parseLong(fields[8], "missing_offset", lineNumber);
            long missingLengthLong = parseLong(fields[9], "missing_length", lineNumber);
            long rowCount = parseLong(fields[12], "row_count", lineNumber);
            long validCount = parseLong(fields[13], "valid_count", lineNumber);
            long missingCount = parseLong(fields[14], "missing_count", lineNumber);
            if (valuesOffset < 0 || missingOffset < 0 || valuesLengthLong < 0 || missingLengthLong < 0
                    || rowCount < 0 || validCount < 0 || missingCount < 0) {
                throw new IllegalArgumentException("negative DBF manifest field at TSV line " + lineNumber);
            }
            if (rowCount > DbpFloat64Codec.MAX_BLOCK_VALUES || validCount > rowCount
                    || missingCount != rowCount - validCount) {
                throw new IllegalArgumentException("invalid DBF block counts at TSV line " + lineNumber);
            }
            if (valuesLengthLong != validCount * width) {
                throw new IllegalArgumentException("values_length mismatch at TSV line " + lineNumber);
            }
            long expectedMissingLength = (rowCount + 7) / 8;
            if (missingLengthLong != expectedMissingLength) {
                throw new IllegalArgumentException("missing_length mismatch at TSV line " + lineNumber);
            }
            if (valuesLengthLong > Integer.MAX_VALUE || missingLengthLong > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("oversized DBF block field at TSV line " + lineNumber);
            }
            requireShaField(fields[6], "values_sha256", lineNumber);
            requireShaField(fields[10], "missing_sha256", lineNumber);
            return new BlockRow(
                    aggregateId,
                    width,
                    maxPoint,
                    fields[3],
                    valuesOffset,
                    (int) valuesLengthLong,
                    fields[6],
                    fields[7],
                    missingOffset,
                    fields[10],
                    rowCount,
                    validCount,
                    missingCount,
                    (int) missingLengthLong);
        }

        private static void requireShaField(String value, String name, int lineNumber) {
            if (value.length() != 64) {
                throw new IllegalArgumentException("invalid " + name + " at TSV line " + lineNumber);
            }
            for (int i = 0; i < value.length(); i++) {
                char c = value.charAt(i);
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                    throw new IllegalArgumentException("invalid " + name + " at TSV line " + lineNumber);
                }
            }
        }

        private static int parseInt(String value, String name, int lineNumber) {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException("invalid " + name + " at TSV line " + lineNumber, error);
            }
        }

        private static long parseLong(String value, String name, int lineNumber) {
            try {
                return Long.parseLong(value);
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException("invalid " + name + " at TSV line " + lineNumber, error);
            }
        }
    }

    private static final class BlockResult {
        final byte[] frame;
        final int compressedBytes;
        final long encodeNanos;
        final long decodeNanos;
        final int mode;
        final int verifiedValues;

        BlockResult(byte[] frame, int verifiedValues, long encodeNanos, long decodeNanos) {
            this.frame = frame;
            this.compressedBytes = frame.length;
            this.encodeNanos = encodeNanos;
            this.decodeNanos = decodeNanos;
            this.mode = DbpFloat64Codec.modeOf(frame);
            this.verifiedValues = verifiedValues;
        }
    }

    private static final class Aggregate {
        final MessageDigest frameSequence = newSha256();
        final String id;
        final int width;
        long blocks;
        long rows;
        long missingCount;
        long validityBytes;
        long validCount;
        long compressedBytes;
        long encodeNanos;
        long decodeNanos;
        long scaledBlocks;
        long rawBlocks;
        long verifiedValues;

        Aggregate(String id, int width) {
            this.id = id;
            this.width = width;
        }

        void add(BlockRow row, BlockResult result) {
            int length = result.frame.length;
            byte[] lengthPrefix = new byte[4];
            for (int b = 0; b < 4; b++) lengthPrefix[b] = (byte) (length >>> (8 * b));
            frameSequence.update(lengthPrefix);
            frameSequence.update(result.frame);
            blocks++;
            rows += row.rowCount;
            missingCount += row.missingCount;
            validityBytes += row.missingLength;
            validCount += row.validCount;
            compressedBytes += result.compressedBytes;
            encodeNanos += result.encodeNanos;
            decodeNanos += result.decodeNanos;
            verifiedValues += result.verifiedValues;
            if (result.mode == DbpFloat64Codec.MODE_SCALED) scaledBlocks++;
            else rawBlocks++;
        }

        String toJson() {
            String status = validCount == 0 ? "n_a" : "ok";
            long originalBytes = validCount * width;
            long headerBytes = blocks * DbpFloat64Codec.HEADER_SIZE;
            return "{\"status\":\"" + status + "\",\"aggregate_id\":\"" + jsonEscape(id)
                    + "\",\"width\":" + width
                    + ",\"blocks\":" + blocks
                    + ",\"rows\":" + rows
                    + ",\"missing_count\":" + missingCount
                    + ",\"validity_bytes\":" + validityBytes
                    + ",\"header_bytes\":" + headerBytes
                    + ",\"valid_count\":" + validCount
                    + ",\"original_bytes\":" + originalBytes
                    + ",\"compressed_bytes\":" + compressedBytes
                    + ",\"codec_wire_bytes\":" + compressedBytes
                    + ",\"frame_compressed_bytes\":" + compressedBytes
                    + ",\"encode_ns\":" + encodeNanos
                    + ",\"decode_ns\":" + decodeNanos
                    + ",\"scaled_blocks\":" + scaledBlocks
                    + ",\"raw_blocks\":" + rawBlocks
                    + ",\"verified_values\":" + verifiedValues
                    + ",\"frame_sequence_sha256\":\"" + hex(frameSequence.digest()) + "\"}";
        }
    }

    private static String jsonEscape(String value) {
        StringBuilder out = new StringBuilder(value.length() + 8);
        for (int i = 0; i < value.length(); i++) {
            char current = value.charAt(i);
            switch (current) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default:
                    if (current < 0x20) out.append(String.format("\\u%04x", (int) current));
                    else out.append(current);
            }
        }
        return out.toString();
    }

    private static final class Arguments {
        final Path manifest;
        final Path aggregates;
        final Path output;

        Arguments(Path manifest, Path aggregates, Path output) {
            this.manifest = manifest;
            this.aggregates = aggregates;
            this.output = output;
        }

        static Arguments parse(String[] args) {
            Path manifest = null, aggregates = null, output = null;
            Integer repeats = null;
            for (int i = 0; i < args.length; i += 2) {
                if (i + 1 >= args.length) throw new IllegalArgumentException("missing value for " + args[i]);
                switch (args[i]) {
                    case "--manifest": manifest = Path.of(args[i + 1]); break;
                    case "--aggregates": aggregates = Path.of(args[i + 1]); break;
                    case "--output": output = Path.of(args[i + 1]); break;
                    case "--repeats": repeats = Integer.parseInt(args[i + 1]); break;
                    default: throw new IllegalArgumentException("unknown bench-manifest flag: " + args[i]);
                }
            }
            if (manifest == null || aggregates == null || output == null || repeats == null) {
                throw new IllegalArgumentException("bench-manifest requires manifest, aggregates, output, repeats");
            }
            if (repeats != 1) throw new IllegalArgumentException("bench-manifest currently requires --repeats 1");
            return new Arguments(manifest, aggregates, output);
        }
    }
}
