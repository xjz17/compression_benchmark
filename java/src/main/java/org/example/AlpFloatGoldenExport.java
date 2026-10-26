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
 * Serializes each float column as ALP windows (same layout C++ golden reader expects):
 * repeat: u8 fac, u8 exp, u16 vecLen LE, u16 excCount LE, u32 nWords LE, nWords×u32 LE, excCount×(u16 idx LE + f32 LE).
 */
public final class AlpFloatGoldenExport {

    private AlpFloatGoldenExport() {}

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
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        ByteBuffer acc = ByteBuffer.allocate(1 << 22).order(ByteOrder.LITTLE_ENDIAN);
        int colN = col.length;
        for (int off = 0; off < colN; ) {
            int vecLen = Math.min(1024, colN - off);
            float[] window = new float[vecLen];
            for (int i = 0; i < vecLen; i++) {
                window[i] = col[off + i];
            }
            int[] fe = AlpFloatBench.findBestFactorExponent(window, vecLen);
            if (fe == null) {
                throw new IllegalStateException("Alp golden: no factor/exponent");
            }
            int fac = fe[0];
            int exp = fe[1];
            int[] encoded = new int[vecLen];
            int[] excPos = new int[vecLen];
            float[] excVal = new float[vecLen];
            int excCount = 0;
            int placeholder = 0;
            boolean foundPh = false;
            for (int i = 0; i < vecLen; i++) {
                float v = window[i];
                int enc = AlpFloatBench.encodeValue(v, fac, exp);
                float dec = AlpFloatBench.decodeValue(enc, fac, exp);
                if (Float.floatToRawIntBits(dec) == Float.floatToRawIntBits(v)) {
                    encoded[i] = enc;
                    if (!foundPh) {
                        placeholder = enc;
                        foundPh = true;
                    }
                } else {
                    excPos[excCount] = i;
                    excVal[excCount] = v;
                    excCount++;
                }
            }
            if (!foundPh) {
                throw new IllegalStateException("Alp golden: no placeholder");
            }
            for (int j = 0; j < excCount; j++) {
                encoded[excPos[j]] = placeholder;
            }
            int[] buf = new int[Math.max(vecLen * 2, vecLen + 4096)];
            IntWrapper inOff = new IntWrapper(0);
            IntWrapper outOff = new IntWrapper(0);
            codec.compress(encoded, inOff, vecLen, buf, outOff);
            int[] compressed = Arrays.copyOf(buf, outOff.get());
            acc.put((byte) fac);
            acc.put((byte) exp);
            acc.putShort((short) vecLen);
            acc.putShort((short) excCount);
            acc.putInt(compressed.length);
            for (int w : compressed) {
                acc.putInt(w);
            }
            for (int j = 0; j < excCount; j++) {
                acc.putShort((short) excPos[j]);
                acc.putFloat(excVal[j]);
            }
            off += vecLen;
        }
        byte[] out = new byte[acc.position()];
        acc.flip();
        acc.get(out);
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: AlpFloatGoldenExport <output_dir>");
            System.exit(2);
        }
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);
        List<ManifestRow> manifest = new ArrayList<>();

        writeGolden(dir, manifest, "alp_float_n64.bin", exportColumn(defaultFloats64()), 64);
        writeGolden(dir, manifest, "alp_float_n512.bin", exportColumn(floats512()), 512);

        writeManifest(dir, manifest);
    }
}
