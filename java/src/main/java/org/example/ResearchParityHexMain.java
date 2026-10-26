package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Emit deterministic hex lines for Python ↔ JAR parity tests.
 *
 * <p>Usage: {@code java -cp ... org.example.ResearchParityHexMain <csv-path> <MODE>}
 *
 * <p>Modes: {@code ELF_ON_CHIMP}, {@code ELF_PLUS}, {@code CAMEL}, {@code CHIMP_FLOAT},
 * {@code SIMPIECE_0P005}, {@code SIMPIECE_0P01} (case-insensitive; normalized to uppercase).
 *
 * <p>Output lines: {@code colIndex chunkIndex hexLowercase} (one chunk per Elf/Camel row;
 * Chimp32 and Sim-Piece use chunkIndex 0 only).
 *
 * <p>CSV: first row header; all columns parsed as {@code float} (matches small fixtures under
 * {@code benchmark/fixtures/}).
 */
public final class ResearchParityHexMain {

    private ResearchParityHexMain() {
    }

    private static String toHex(byte[] a) {
        char[] hex = "0123456789abcdef".toCharArray();
        StringBuilder sb = new StringBuilder(a.length * 2);
        for (byte x : a) {
            int v = x & 0xff;
            sb.append(hex[v >>> 4]).append(hex[v & 0xf]);
        }
        return sb.toString();
    }

    static List<ArrayList<Float>> readFloatTable(Path csv) throws Exception {
        List<ArrayList<Float>> columns = new ArrayList<>();
        try (Reader r = Files.newBufferedReader(csv)) {
            CSVParser parser =
                    CSVFormat.DEFAULT
                            .builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .build()
                            .parse(r);
            List<String> hdr = parser.getHeaderNames();
            if (hdr == null || hdr.isEmpty()) {
                throw new IllegalArgumentException("CSV needs a header row");
            }
            int nc = hdr.size();
            for (int i = 0; i < nc; i++) {
                columns.add(new ArrayList<>());
            }
            for (CSVRecord rec : parser) {
                for (int i = 0; i < nc; i++) {
                    String s = rec.get(i);
                    if (ExperimentDataPolicy.isNaNCell(s)) continue;
                    float value = Float.parseFloat(s.trim());
                    if (ExperimentDataPolicy.keepNumericValue(value)) columns.get(i).add(value);
                }
            }
        }
        return columns;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println(
                    "Usage: java -cp <classpath> org.example.ResearchParityHexMain <csv> <MODE>\n"
                            + "MODE: ELF_ON_CHIMP | ELF_PLUS | CAMEL | CHIMP_FLOAT | SIMPIECE_0P005 | SIMPIECE_0P01");
            System.exit(2);
        }
        Path csv = Path.of(args[0]);
        String mode = args[1].trim().toUpperCase(Locale.ROOT);
        List<ArrayList<Float>> columns = readFloatTable(csv);
        for (int ci = 0; ci < columns.size(); ci++) {
            ArrayList<Float> col = columns.get(ci);
            switch (mode) {
                case "ELF_ON_CHIMP": {
                    List<byte[]> chunks = ExternalCompressionBench.parityCompressedChunksElfOnChimp(col);
                    for (int k = 0; k < chunks.size(); k++) {
                        System.out.println(ci + " " + k + " " + toHex(chunks.get(k)));
                    }
                    break;
                }
                case "ELF_PLUS": {
                    List<byte[]> chunks = ExternalCompressionBench.parityCompressedChunksElfPlus(col);
                    for (int k = 0; k < chunks.size(); k++) {
                        System.out.println(ci + " " + k + " " + toHex(chunks.get(k)));
                    }
                    break;
                }
                case "CAMEL": {
                    List<byte[]> chunks = ExternalCompressionBench.parityCompressedChunksCamel(col);
                    for (int k = 0; k < chunks.size(); k++) {
                        System.out.println(ci + " " + k + " " + toHex(chunks.get(k)));
                    }
                    break;
                }
                case "CHIMP_FLOAT": {
                    byte[] one = ExternalCompressionBench.parityCompressedChimp32FloatWholeColumn(col);
                    System.out.println(ci + " 0 " + toHex(one));
                    break;
                }
                case "SIMPIECE_0P005": {
                    byte[] p = ExternalCompressionBench.parityCompressedSimPiece(col, 0.005);
                    System.out.println(ci + " 0 " + toHex(p));
                    break;
                }
                case "SIMPIECE_0P01": {
                    byte[] p = ExternalCompressionBench.parityCompressedSimPiece(col, 0.01);
                    System.out.println(ci + " 0 " + toHex(p));
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unknown MODE: " + args[1]);
            }
        }
    }
}
