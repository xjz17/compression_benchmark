package org.example;

import java.io.BufferedWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import me.lemire.integercompression.Composition;
import me.lemire.integercompression.FastPFOR;
import me.lemire.integercompression.IntWrapper;
import me.lemire.integercompression.IntegerCODEC;
import me.lemire.integercompression.VariableByte;

/**
 * Exports one column per file: LE int32 {@code p} (decimal scale) + LE int32 FastPFOR+VB words for scaled ints.
 */
public final class LeCoFloatGoldenExport {

    private LeCoFloatGoldenExport() {}

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
        Files.write(dir.resolve(name), data);
        manifest.add(new ManifestRow(name, valuesLen, data.length));
    }

    private static float[] defaultFloats64() {
        float[] v = new float[64];
        for (int i = 0; i < 64; i++) {
            v[i] = 1.0f + i * 0.03125f;
        }
        return v;
    }

    private static float[] floats512() {
        float[] v = new float[512];
        for (int i = 0; i < 512; i++) {
            v[i] = 1.25f + i * 0.0625f;
        }
        return v;
    }

    private static byte[] exportColumn(float[] col) {
        int n = col.length;
        int colScale = 0;
        for (float v : col) {
            int s = ExternalCompressionBench.minDecimalScaleExactFloat(v);
            if (s < 0) {
                throw new IllegalStateException("LeCo golden: non-encodable float");
            }
            if (s > colScale) {
                colScale = s;
            }
        }
        double pow = Math.pow(10.0, colScale);
        int[] data = new int[n];
        for (int i = 0; i < n; i++) {
            float v = col[i];
            long m = Math.round((double) v * pow);
            int im = (int) m;
            float r = (float) ((double) im / pow);
            if (Float.floatToRawIntBits(r) != Float.floatToRawIntBits(v)) {
                throw new IllegalStateException("LeCo golden: round-trip at " + i);
            }
            data[i] = im;
        }
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        int[] buf = new int[Math.max(n * 2, n + 4096)];
        IntWrapper in = new IntWrapper(0);
        IntWrapper out = new IntWrapper(0);
        codec.compress(data, in, n, buf, out);
        int[] compressed = Arrays.copyOf(buf, out.get());
        ByteBuffer bb = ByteBuffer.allocate(4 + compressed.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(colScale);
        for (int w : compressed) {
            bb.putInt(w);
        }
        return bb.array();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: LeCoFloatGoldenExport <output_dir>");
            System.exit(2);
        }
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);
        List<ManifestRow> manifest = new ArrayList<>();

        float[] n64 = defaultFloats64();
        writeGolden(dir, manifest, "leco_float_n64.bin", exportColumn(n64), n64.length);
        float[] n512 = floats512();
        writeGolden(dir, manifest, "leco_float_n512.bin", exportColumn(n512), n512.length);

        writeManifest(dir, manifest);
    }
}
