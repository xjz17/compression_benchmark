package org.example;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Exports ELF-on-Chimp32 and Elf+ (XOR32) float column bytes matching
 * {@link ExternalCompressionBench#benchElfVariant32Float} (2048-value chunks, {@code writtenBits}
 * truncation).
 *
 * <p>Requires {@code external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar} plus
 * {@code target/dependency/*.jar} (run {@code mvn dependency:copy-dependencies} in that module).
 */
public final class ElfFloatGoldenExport {

    private ElfFloatGoldenExport() {}

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

    private static byte[] exportElfColumn(float[] col, String encoderClass, String xorBackendField) throws Exception {
        java.io.File elfJar = ExternalCompressionBench.resolveElfJarForIsolatedBench();
        if (elfJar == null) {
            throw new IllegalStateException("ELF JAR not found (build external/compression/elf with mvn package)");
        }
        List<URL> urlList = new ArrayList<>();
        try {
            urlList.add(elfJar.toURI().toURL());
            Path depDir = elfJar.toPath().getParent().resolve("dependency");
            if (Files.isDirectory(depDir)) {
                try (DirectoryStream<Path> ds = Files.newDirectoryStream(depDir, "*.jar")) {
                    for (Path p : ds) {
                        urlList.add(p.toUri().toURL());
                    }
                }
            }
        } catch (MalformedURLException e) {
            throw new IllegalStateException(e);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        URLClassLoader cl =
                new URLClassLoader(urlList.toArray(new URL[0]), ClassLoader.getSystemClassLoader());
        try {
            Class<?> encCl = Class.forName(encoderClass, true, cl);
            Constructor<?> encCtor = encCl.getConstructor();
            Method addValue = encCl.getMethod("addValue", float.class);
            Method close = encCl.getMethod("close");
            Method getBytes = encCl.getMethod("getBytes");
            Field xorField = encCl.getDeclaredField(xorBackendField);
            xorField.setAccessible(true);

            final int elfChunkPoints = 2048;
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            int n = col.length;
            for (int start = 0; start < n; start += elfChunkPoints) {
                int end = Math.min(n, start + elfChunkPoints);
                Object compressor = encCtor.newInstance();
                for (int i = start; i < end; i++) {
                    addValue.invoke(compressor, col[i]);
                }
                close.invoke(compressor);
                Object xor32 = xorField.get(compressor);
                Method getOutputStream = xor32.getClass().getMethod("getOutputStream");
                Object bitStream = getOutputStream.invoke(xor32);
                Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                int compBytes = (int) ((wb + 7L) / 8L);
                byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                bos.write(rawBuf, 0, Math.min(compBytes, rawBuf.length));
            }
            return bos.toByteArray();
        } finally {
            cl.close();
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: ElfFloatGoldenExport <golden_root_dir>");
            System.err.println("  writes elf_on_chimp_float/ and elf_plus_float/ under that path");
            System.exit(2);
        }
        Path goldenRoot = Paths.get(args[0]);
        Path dirChimp = goldenRoot.resolve("elf_on_chimp_float");
        Path dirPlus = goldenRoot.resolve("elf_plus_float");
        List<ManifestRow> manChimp = new ArrayList<>();
        List<ManifestRow> manPlus = new ArrayList<>();

        float[] n64 = defaultFloats64();
        float[] n512 = floats512();

        writeGolden(
                dirChimp,
                manChimp,
                "elf_on_chimp_float_n64.bin",
                exportElfColumn(
                        n64,
                        "org.urbcomp.startdb.compress.elf.compressor32.ElfOnChimpCompressor32",
                        "chimp32"),
                n64.length);
        writeGolden(
                dirChimp,
                manChimp,
                "elf_on_chimp_float_n512.bin",
                exportElfColumn(
                        n512,
                        "org.urbcomp.startdb.compress.elf.compressor32.ElfOnChimpCompressor32",
                        "chimp32"),
                n512.length);
        writeManifest(dirChimp, manChimp);

        writeGolden(
                dirPlus,
                manPlus,
                "elf_plus_float_n64.bin",
                exportElfColumn(
                        n64,
                        "org.urbcomp.startdb.compress.elf.compressor32.ElfCompressor32",
                        "xorCompressor32"),
                n64.length);
        writeGolden(
                dirPlus,
                manPlus,
                "elf_plus_float_n512.bin",
                exportElfColumn(
                        n512,
                        "org.urbcomp.startdb.compress.elf.compressor32.ElfCompressor32",
                        "xorCompressor32"),
                n512.length);
        writeManifest(dirPlus, manPlus);

        System.out.println("ELF float goldens -> " + dirChimp + " and " + dirPlus);
    }
}
