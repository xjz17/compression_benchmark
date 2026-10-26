package org.example;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Exports Sim-Piece float blobs matching {@link ExternalCompressionBench#parityCompressedSimPiece}.
 *
 * <p>Uses in-repo {@link SimPieceFloatNative} via {@code benchmark-java-*.jar} only.
 */
public final class SimPieceFloatGoldenExport {

    private SimPieceFloatGoldenExport() {}

    private static final class ManifestRow {
        final String filename;
        final int valuesLen;
        final double relRangeFrac;
        final int bytesLen;

        ManifestRow(String filename, int valuesLen, double relRangeFrac, int bytesLen) {
            this.filename = filename;
            this.valuesLen = valuesLen;
            this.relRangeFrac = relRangeFrac;
            this.bytesLen = bytesLen;
        }
    }

    private static void writeManifest(Path dir, List<ManifestRow> rows) throws Exception {
        rows.sort(Comparator.comparing(r -> r.filename));
        Path mf = dir.resolve("manifest.tsv");
        try (BufferedWriter w = Files.newBufferedWriter(mf)) {
            w.write("filename\tvalues_len\trel_range_frac\tbytes_len\n");
            for (ManifestRow r : rows) {
                w.write(
                        r.filename + '\t' + r.valuesLen + '\t' + r.relRangeFrac + '\t' + r.bytesLen + '\n');
            }
        }
    }

    private static void writeGolden(
            Path dir, List<ManifestRow> manifest, String name, double rel, byte[] data, int valuesLen)
            throws Exception {
        Files.createDirectories(dir);
        Files.write(dir.resolve(name), data);
        manifest.add(new ManifestRow(name, valuesLen, rel, data.length));
    }

    private static float[] camelStableFloats64() {
        float[] v = new float[64];
        for (int i = 0; i < 64; i++) {
            v[i] = 1.0f + i;
        }
        return v;
    }

    private static float[] camelStableFloats512() {
        float[] v = new float[512];
        for (int i = 0; i < 512; i++) {
            v[i] = 100.0f + i * 0.5f;
        }
        return v;
    }

    private static byte[] export(ArrayList<Float> col, double rel) throws Exception {
        return ExternalCompressionBench.parityCompressedSimPiece(col, rel);
    }

    private static ArrayList<Float> toList(float[] v) {
        ArrayList<Float> a = new ArrayList<>(v.length);
        for (float f : v) {
            a.add(f);
        }
        return a;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: SimPieceFloatGoldenExport <golden_root_dir>");
            System.err.println("  writes simpiece_float/ under that path");
            System.exit(2);
        }
        Path goldenRoot = Paths.get(args[0]);
        Path dir = goldenRoot.resolve("simpiece_float");
        List<ManifestRow> manifest = new ArrayList<>();

        float[] n64 = camelStableFloats64();
        float[] n512 = camelStableFloats512();
        final double r0 = 0.005;
        final double r1 = 0.01;

        writeGolden(dir, manifest, "simpiece_float_relRange_0p005_n64.bin", r0, export(toList(n64), r0), n64.length);
        writeGolden(
                dir, manifest, "simpiece_float_relRange_0p005_n512.bin", r0, export(toList(n512), r0), n512.length);
        writeGolden(dir, manifest, "simpiece_float_relRange_0p01_n64.bin", r1, export(toList(n64), r1), n64.length);
        writeGolden(
                dir, manifest, "simpiece_float_relRange_0p01_n512.bin", r1, export(toList(n512), r1), n512.length);
        writeManifest(dir, manifest);
        System.out.println("Sim-Piece float goldens -> " + dir);
    }
}
