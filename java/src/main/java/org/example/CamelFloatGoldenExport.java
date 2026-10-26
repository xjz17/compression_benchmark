package org.example;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Exports Camel float column bytes matching {@link ExternalCompressionBench#benchCamelFloat} (1000-value
 * chunks, compressed length from {@code writtenBits} truncation).
 *
 * <p>Requires {@code external/compression/camel/target/start-compress-1.0-SNAPSHOT.jar} and
 * {@code target/dependency/*.jar} from {@code mvn -DskipTests package} in that module.
 */
public final class CamelFloatGoldenExport {

    private CamelFloatGoldenExport() {}

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

    private static void writeGolden(Path dir, List<ManifestRow> manifest, String name, byte[] data, int valuesLen)
            throws Exception {
        Files.createDirectories(dir);
        Files.write(dir.resolve(name), data);
        manifest.add(new ManifestRow(name, valuesLen, data.length));
    }

    /**
     * Camel JAR lossless path does not round-trip every float ramp (e.g. {@code 1.0f, 1.03125f} fails
     * {@link ExternalCompressionBench#parityCompressedChunksCamel}); use decimals Camel handles on
     * golden columns.
     */
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

    private static byte[] exportColumn(float[] col) throws Exception {
        ArrayList<Float> al = new ArrayList<>(col.length);
        for (float f : col) {
            al.add(f);
        }
        ArrayList<byte[]> chunks = ExternalCompressionBench.parityCompressedChunksCamel(al);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        for (byte[] c : chunks) {
            bos.write(c);
        }
        return bos.toByteArray();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: CamelFloatGoldenExport <golden_root_dir>");
            System.err.println("  writes camel_float/ under that path");
            System.exit(2);
        }
        Path goldenRoot = Paths.get(args[0]);
        Path dir = goldenRoot.resolve("camel_float");
        List<ManifestRow> manifest = new ArrayList<>();

        float[] n64 = camelStableFloats64();
        float[] n512 = camelStableFloats512();
        writeGolden(dir, manifest, "camel_float_n64.bin", exportColumn(n64), n64.length);
        writeGolden(dir, manifest, "camel_float_n512.bin", exportColumn(n512), n512.length);
        writeManifest(dir, manifest);
        System.out.println("Camel float goldens -> " + dir);
    }
}
