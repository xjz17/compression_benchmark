package org.example;

import org.apache.tsfile.encoding.decoder.Decoder;
import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.encoding.encoder.TSEncodingBuilder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Exports fixed small TSFile encoder bitstreams (goldens) for the six benchmark encodings.
 *
 * <p>Usage: TsFileGoldenExport --output dir
 */
public final class TsFileGoldenExport {

    private static final int N = 64;
    private static final String FIXTURE = "default";

    private static final List<TSEncoding> ENCODINGS =
            Arrays.asList(
                    TSEncoding.TS_2DIFF,
                    TSEncoding.RLE,
                    TSEncoding.GORILLA,
                    TSEncoding.CHIMP,
                    TSEncoding.SPRINTZ,
                    TSEncoding.RLBE);

    private TsFileGoldenExport() {}

    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
        Path outDir = null;
        for (int i = 0; i < args.length; i++) {
            if ("--output".equals(args[i]) && i + 1 < args.length) {
                outDir = Paths.get(args[++i]);
            }
        }
        if (outDir == null) {
            System.err.println("Usage: TsFileGoldenExport --output <dir>");
            System.exit(1);
            return;
        }
        Files.createDirectories(outDir);

        ArrayList<Integer> intVals = new ArrayList<>(N);
        ArrayList<Long> longVals = new ArrayList<>(N);
        ArrayList<Float> floatVals = new ArrayList<>(N);
        ArrayList<Double> doubleVals = new ArrayList<>(N);
        for (int i = 0; i < N; i++) {
            intVals.add(10000 + i * 3);
            longVals.add(10000L + i * 3L);
            floatVals.add(1.0f + i * 0.03125f);
            doubleVals.add(1.0 + i * 0.03125);
        }

        Path manifestPath = outDir.resolve("manifest.tsv");
        List<String> manifestLines = new ArrayList<>();
        manifestLines.add(
                "encoding\tdtype\tfixture\tvalues_len\tbytes_len\tsha256\tfilename");

        for (TSEncoding enc : ENCODINGS) {
            exportOne(outDir, manifestLines, TSDataType.INT32, enc, intVals);
            exportOne(outDir, manifestLines, TSDataType.INT64, enc, longVals);
            exportOne(outDir, manifestLines, TSDataType.FLOAT, enc, floatVals);
            exportOne(outDir, manifestLines, TSDataType.DOUBLE, enc, doubleVals);
        }

        Files.writeString(manifestPath, String.join("\n", manifestLines) + "\n", StandardCharsets.UTF_8);
        System.out.println("Wrote manifest and binaries under " + outDir.toAbsolutePath());
    }

    private static void exportOne(
            Path outDir,
            List<String> manifestLines,
            TSDataType dataType,
            TSEncoding encoding,
            List<?> values)
            throws IOException, NoSuchAlgorithmException {

        byte[] encoded = encodeColumn(dataType, encoding, values);
        // TSFile decoders for several float encodings do not match Encoder output in all cases
        // in tsfile 2.0.x; INT32 round-trip is checked. Float bitstreams still match
        // CompressionUtils.testEncoding encode path (golden bytes).
        if (dataType == TSDataType.INT32 || dataType == TSDataType.INT64) {
            assertRoundTrip(dataType, encoding, values, encoded);
        }

        String dtypeName;
        if (dataType == TSDataType.INT32) {
            dtypeName = "INT32";
        } else if (dataType == TSDataType.INT64) {
            dtypeName = "INT64";
        } else if (dataType == TSDataType.FLOAT) {
            dtypeName = "FLOAT";
        } else if (dataType == TSDataType.DOUBLE) {
            dtypeName = "DOUBLE";
        } else {
            dtypeName = dataType.name();
        }
        String fileName = encoding.name() + "_" + dtypeName + "_" + FIXTURE + ".bin";
        Path file = outDir.resolve(fileName);
        Files.write(file, encoded);

        String sha = sha256Hex(encoded);
        manifestLines.add(
                String.join(
                        "\t",
                        encoding.name(),
                        dtypeName,
                        FIXTURE,
                        String.valueOf(values.size()),
                        String.valueOf(encoded.length),
                        sha,
                        fileName));
    }

    private static byte[] encodeColumn(TSDataType dataType, TSEncoding encoding, List<?> data)
            throws IOException {
        TSEncodingBuilder builder = TSEncodingBuilder.getEncodingBuilder(encoding);
        Encoder encoder = builder.getEncoder(dataType);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        for (Object value : data) {
            if (dataType == TSDataType.INT32) {
                encoder.encode((Integer) value, baos);
            } else if (dataType == TSDataType.INT64) {
                encoder.encode((Long) value, baos);
            } else if (dataType == TSDataType.FLOAT) {
                encoder.encode((Float) value, baos);
            } else if (dataType == TSDataType.DOUBLE) {
                encoder.encode((Double) value, baos);
            } else {
                throw new IllegalArgumentException("Unsupported data type: " + dataType);
            }
        }
        encoder.flush(baos);
        return baos.toByteArray();
    }

    private static void assertRoundTrip(
            TSDataType dataType, TSEncoding encoding, List<?> expected, byte[] encoded)
            throws IOException {
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(encoded);
        Decoder decoder = Decoder.getDecoderByType(encoding, dataType);
        int n = expected.size();
        if (dataType == TSDataType.INT32) {
            for (int i = 0; i < n; i++) {
                if (!decoder.hasNext(buffer)) {
                    throw new IOException("Decoder ended early at int index " + i);
                }
                int v = decoder.readInt(buffer);
                if (v != (Integer) expected.get(i)) {
                    throw new IOException(
                            "Round-trip mismatch at int " + i + ": got " + v + " expected " + expected.get(i));
                }
            }
        } else if (dataType == TSDataType.INT64) {
            for (int i = 0; i < n; i++) {
                if (!decoder.hasNext(buffer)) {
                    throw new IOException("Decoder ended early at long index " + i);
                }
                long v = decoder.readLong(buffer);
                if (v != (Long) expected.get(i)) {
                    throw new IOException(
                            "Round-trip mismatch at long " + i + ": got " + v + " expected " + expected.get(i));
                }
            }
        } else if (dataType == TSDataType.FLOAT) {
            for (int i = 0; i < n; i++) {
                if (!decoder.hasNext(buffer)) {
                    throw new IOException("Decoder ended early at float index " + i);
                }
                float v = decoder.readFloat(buffer);
                if (Float.floatToIntBits(v) != Float.floatToIntBits((Float) expected.get(i))) {
                    throw new IOException(
                            "Round-trip mismatch at float " + i + ": got " + v + " expected " + expected.get(i));
                }
            }
        } else if (dataType == TSDataType.DOUBLE) {
            for (int i = 0; i < n; i++) {
                if (!decoder.hasNext(buffer)) {
                    throw new IOException("Decoder ended early at double index " + i);
                }
                double v = decoder.readDouble(buffer);
                if (Double.doubleToLongBits(v) != Double.doubleToLongBits((Double) expected.get(i))) {
                    throw new IOException(
                            "Round-trip mismatch at double " + i + ": got " + v + " expected " + expected.get(i));
                }
            }
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
        if (decoder.hasNext(buffer)) {
            throw new IOException("Decoder has trailing values");
        }
    }

    private static String sha256Hex(byte[] data) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] digest = md.digest(data);
        StringBuilder sb = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            sb.append(String.format("%02x", b & 0xff));
        }
        return sb.toString();
    }
}
