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
 * Exports compressed streams for {@link ExternalCompressionBench#benchJavaFastPFOR} /
 * {@link ExternalCompressionBench#benchJavaFastPFORFloat} (Composition(FastPFOR, VariableByte)).
 * Golden layout: little-endian int32 words (same packing as Java's {@code int[]} compressed buffer).
 */
public final class FastPFORGoldenExport {

    private FastPFORGoldenExport() {}

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

    /** Same 64 ints as {@link ColumnarGoldenExport} default int column. */
    private static int[] defaultInts64() {
        int[] v = new int[64];
        for (int i = 0; i < 64; i++) {
            v[i] = 10000 + i * 3;
        }
        return v;
    }

    private static float[] defaultFloats64() {
        float[] v = new float[64];
        for (int i = 0; i < 64; i++) {
            v[i] = 1.0f + i * 0.03125f;
        }
        return v;
    }

    private static byte[] compressIntWords(int[] data) {
        int n = data.length;
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        int[] compressed = new int[Math.max(n * 2, n + 4096)];
        IntWrapper inOff = new IntWrapper(0);
        IntWrapper outOff = new IntWrapper(0);
        codec.compress(data, inOff, n, compressed, outOff);
        int outWords = outOff.get();
        compressed = Arrays.copyOf(compressed, outWords);
        ByteBuffer bb = ByteBuffer.allocate(compressed.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int w : compressed) {
            bb.putInt(w);
        }
        return bb.array();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: FastPFORGoldenExport <output_dir>");
            System.exit(2);
        }
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);
        List<ManifestRow> manifest = new ArrayList<>();

        int[] n64 = defaultInts64();
        writeGolden(dir, manifest, "fastpfor_int_n64.bin", compressIntWords(n64), n64.length);

        float[] f64 = defaultFloats64();
        int[] bits = new int[f64.length];
        for (int i = 0; i < f64.length; i++) {
            bits[i] = Float.floatToRawIntBits(f64[i]);
        }
        writeGolden(dir, manifest, "fastpfor_float_n64.bin", compressIntWords(bits), f64.length);

        int[] n512 = new int[512];
        for (int i = 0; i < 512; i++) {
            n512[i] = 1000 + i;
        }
        writeGolden(dir, manifest, "fastpfor_int_n512.bin", compressIntWords(n512), n512.length);

        int[] f512bits = new int[512];
        for (int i = 0; i < 512; i++) {
            f512bits[i] = Float.floatToRawIntBits(1.25f + i * 0.0625f);
        }
        writeGolden(dir, manifest, "fastpfor_float_n512.bin", compressIntWords(f512bits), 512);

        writeManifest(dir, manifest);
    }
}
