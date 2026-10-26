package org.example;

import java.io.BufferedWriter;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Exports Chimp32 compressed bytes matching {@link ExternalCompressionBench#benchChimp32Paper}
 * (truncation from {@code getSize()}, not full {@code OutputBitStream#writtenBits()}).
 * Uses reflection so {@code benchmark-java} does not need a compile-time dependency on Chimp classes.
 */
public final class Chimp32GoldenExport {

    private Chimp32GoldenExport() {}

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

    /**
     * Same truncation as {@code benchChimp32Paper}: {@code Arrays.copyOf(getOut(), (getSize() + 7) / 8)}.
     */
    private static byte[] compressFloatColumn(float[] data) throws Exception {
        Class<?> chimpCl = Class.forName("gr.aueb.delorean.chimp.Chimp32");
        Method addValue = chimpCl.getMethod("addValue", float.class);
        Method close = chimpCl.getMethod("close");
        Method getOut = chimpCl.getMethod("getOut");
        Method getSize = chimpCl.getMethod("getSize");
        Constructor<?> ctor = chimpCl.getConstructor();
        Object compressor = ctor.newInstance();
        for (float f : data) {
            addValue.invoke(compressor, f);
        }
        close.invoke(compressor);
        int bits = (Integer) getSize.invoke(compressor);
        int compBytes = (int) (((long) bits + 7L) / 8L);
        byte[] rawBuf = (byte[]) getOut.invoke(compressor);
        return Arrays.copyOf(rawBuf, compBytes);
    }

    private static byte[] compressIntColumnAsFloatBits(int[] data) throws Exception {
        Class<?> chimpCl = Class.forName("gr.aueb.delorean.chimp.Chimp32");
        Method addValue = chimpCl.getMethod("addValue", float.class);
        Method close = chimpCl.getMethod("close");
        Method getOut = chimpCl.getMethod("getOut");
        Method getSize = chimpCl.getMethod("getSize");
        Constructor<?> ctor = chimpCl.getConstructor();
        Object compressor = ctor.newInstance();
        for (int v : data) {
            addValue.invoke(compressor, Float.intBitsToFloat(v));
        }
        close.invoke(compressor);
        int bits = (Integer) getSize.invoke(compressor);
        int compBytes = (int) (((long) bits + 7L) / 8L);
        byte[] rawBuf = (byte[]) getOut.invoke(compressor);
        return Arrays.copyOf(rawBuf, compBytes);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: Chimp32GoldenExport <output_dir>");
            System.exit(2);
        }
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);
        List<ManifestRow> manifest = new ArrayList<>();

        int[] n64 = defaultInts64();
        writeGolden(dir, manifest, "chimp32_int_n64.bin", compressIntColumnAsFloatBits(n64), n64.length);

        float[] f64 = defaultFloats64();
        writeGolden(dir, manifest, "chimp32_float_n64.bin", compressFloatColumn(f64), f64.length);

        int[] n512 = new int[512];
        for (int i = 0; i < 512; i++) {
            n512[i] = 1000 + i;
        }
        writeGolden(dir, manifest, "chimp32_int_n512.bin", compressIntColumnAsFloatBits(n512), n512.length);

        int[] f512bits = new int[512];
        for (int i = 0; i < 512; i++) {
            f512bits[i] = Float.floatToRawIntBits(1.25f + i * 0.0625f);
        }
        writeGolden(dir, manifest, "chimp32_float_n512.bin", compressIntColumnAsFloatBits(f512bits), 512);

        writeManifest(dir, manifest);
    }
}
