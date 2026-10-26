package org.example;

import java.io.BufferedWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Writes columnar encoding goldens for C++ tests (see benchmark/reference/golden/column_encodings/).
 * Always rewrites {@code manifest.tsv} (filename, values_len, bytes_len) from the written blobs.
 */
public final class ColumnarGoldenExport {

    private ColumnarGoldenExport() {}

    private static final class ManifestRow {
        final String filename;
        final int valuesLen;
        final int bytesLen;

        ManifestRow(String filename, int valuesLen, int bytesLen) {
            this.filename = filename;
            this.valuesLen = valuesLen;
            this.bytesLen = bytesLen;
        }
    }

    private static void writeGolden(Path dir, List<ManifestRow> manifest, String name, byte[] data, int valuesLen)
            throws Exception {
        Files.write(dir.resolve(name), data);
        manifest.add(new ManifestRow(name, valuesLen, data.length));
    }

    private static List<Float> defaultFloats64() {
        List<Float> floats = new ArrayList<>(64);
        for (int i = 0; i < 64; i++) {
            floats.add(1.0f + i * 0.03125f);
        }
        return floats;
    }

    private static List<Integer> defaultInts64() {
        List<Integer> ints = new ArrayList<>(64);
        for (int i = 0; i < 64; i++) {
            ints.add(10000 + i * 3);
        }
        return ints;
    }

    private static void writeManifest(Path dir, List<ManifestRow> rows) throws Exception {
        rows.sort(Comparator.comparing(r -> r.filename));
        Path mf = dir.resolve("manifest.tsv");
        try (BufferedWriter w = Files.newBufferedWriter(mf)) {
            w.write("filename\tvalues_len\tbytes_len\n");
            for (ManifestRow r : rows) {
                w.write(r.filename + '\t' + r.valuesLen + '\t' + r.bytesLen + '\n');
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: ColumnarGoldenExport <output_dir>");
            System.exit(2);
        }
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);

        List<ManifestRow> manifest = new ArrayList<>();

        List<Float> floats = defaultFloats64();
        List<Integer> ints = defaultInts64();

        writeGolden(dir, manifest, "col_dictionary_float_default.bin",
                ColumnarEncodings.encodeDictionaryFloat(floats), floats.size());
        writeGolden(dir, manifest, "delta_binary_packed_int32_default.bin",
                ColumnarEncodings.encodeDeltaZigZagVarint(ints), ints.size());
        writeGolden(dir, manifest, "col_dictionary_int32_default.bin",
                ColumnarEncodings.encodeDictionaryInt(ints), ints.size());
        writeGolden(dir, manifest, "col_rle_int32_default.bin", ColumnarEncodings.encodeRleInt(ints), ints.size());
        writeGolden(dir, manifest, "col_bss_lz4_float_default.bin",
                ColumnarEncodings.encodeBssLz4Float(floats), floats.size());

        ByteBuffer bb = ByteBuffer.allocate(floats.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats) {
            bb.putFloat(f);
        }
        writeGolden(dir, manifest, "col_bss_planar_float_default.bin",
                ColumnarEncodings.byteStreamSplitFloat(bb.array()), floats.size());

        writeGolden(dir, manifest, "delta_binary_packed_empty.bin",
                ColumnarEncodings.encodeDeltaZigZagVarint(List.of()), 0);
        writeGolden(dir, manifest, "col_dictionary_int32_empty.bin",
                ColumnarEncodings.encodeDictionaryInt(List.of()), 0);
        writeGolden(dir, manifest, "col_dictionary_float_empty.bin",
                ColumnarEncodings.encodeDictionaryFloat(List.of()), 0);
        writeGolden(dir, manifest, "col_rle_int32_empty.bin", ColumnarEncodings.encodeRleInt(List.of()), 0);

        List<Integer> dupInts = List.of(10, 20, 10, 20, 30);
        writeGolden(dir, manifest, "col_dictionary_int32_dupes.bin",
                ColumnarEncodings.encodeDictionaryInt(dupInts), dupInts.size());

        List<Float> dupFloats = List.of(1.0f, 1.0f, 2.0f);
        writeGolden(dir, manifest, "col_dictionary_float_dupes.bin",
                ColumnarEncodings.encodeDictionaryFloat(dupFloats), dupFloats.size());

        List<Integer> runs = List.of(42, 42, 42, 7, 7);
        writeGolden(dir, manifest, "col_rle_int32_runs.bin", ColumnarEncodings.encodeRleInt(runs), runs.size());

        List<Integer> neg = List.of(-10, -13, -7, -20);
        writeGolden(dir, manifest, "delta_binary_packed_neg.bin",
                ColumnarEncodings.encodeDeltaZigZagVarint(neg), neg.size());

        writeGolden(dir, manifest, "col_bss_planar_float_empty.bin", new byte[0], 0);
        writeGolden(dir, manifest, "col_bss_lz4_float_empty.bin",
                ColumnarEncodings.encodeBssLz4Float(List.of()), 0);

        writeManifest(dir, manifest);
        System.err.println("Wrote " + manifest.size() + " goldens + manifest.tsv under " + dir.toAbsolutePath());
    }
}
