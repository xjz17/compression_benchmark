package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.tsfile.enums.TSDataType;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Appends rows that still depend on the Java research JAR stack when the primary driver is native
 * {@code benchmark_cpp}. TSFile encodings, FastPFOR, Chimp32, LeCo, ALP, and ELF float variants are
 * native in C++; {@link ExternalCompressionBench#printRows} skips those rows when the hybrid launcher
 * sets {@code WEB_COMPRESSION_CPP_NATIVE_*} env vars. This companion runs {@code printRows} in
 * {@code external-only} phase for any future JAR-only rows (default hybrid env leaves no duplicate float
 * research rows).
 *
 * <p>CSV input is parsed with Apache Commons CSV (RFC 4180), matching {@code pandas.to_csv} / the
 * {@code csv2} reader used in {@code benchmark_cpp}. Naive {@code String.split(",")} breaks on quoted
 * fields and on {@code ""} empty cells, which caused {@link NumberFormatException} on some datasets.
 *
 * <p>Invoked from {@code scripts/cpp_java_companion.py} with the same classpath as
 * {@link Main}. Native C++ handles generic byte compressors, column encodings, SZ/SZ3/Serf CLIs.
 */
public final class CppBenchmarkCompanion {

    private CppBenchmarkCompanion() {}

    public static void main(String[] args) throws IOException {
        String inputPath = null;
        String dataType = null;
        String appendPath = null;
        String phase = "external-only";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--input":
                    if (i + 1 < args.length) {
                        inputPath = args[++i];
                    }
                    break;
                case "--data-type":
                    if (i + 1 < args.length) {
                        dataType = args[++i];
                    }
                    break;
                case "--append":
                    if (i + 1 < args.length) {
                        appendPath = args[++i];
                    }
                    break;
                case "--phase":
                    if (i + 1 < args.length) {
                        phase = args[++i].trim();
                    }
                    break;
                default:
                    break;
            }
        }

        if (inputPath == null || dataType == null || appendPath == null) {
            System.err.println(
                    "Usage: CppBenchmarkCompanion --input <csv> --data-type int|float --append <report.csv> "
                            + "[--phase external-only]");
            System.exit(1);
        }

        File out = new File(appendPath);
        File parent = out.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        TSDataType tsType;
        if ("int".equalsIgnoreCase(dataType)) {
            tsType = TSDataType.INT32;
        } else if ("float".equalsIgnoreCase(dataType)) {
            tsType = TSDataType.FLOAT;
        } else {
            System.err.println("Unsupported data type: " + dataType);
            System.exit(1);
            return;
        }

        ArrayList<byte[]> columnByteData = new ArrayList<>();
        ArrayList<ArrayList<Integer>> columnIntData = new ArrayList<>();
        ArrayList<ArrayList<Float>> columnFloatData = new ArrayList<>();

        try (Reader reader = Files.newBufferedReader(Paths.get(inputPath), StandardCharsets.UTF_8);
                CSVParser parser =
                        CSVFormat.DEFAULT.builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(true)
                                .setTrim(true)
                                .build()
                                .parse(reader)) {
            List<String> headerNames = parser.getHeaderNames();
            if (headerNames == null || headerNames.isEmpty()) {
                System.err.println("CSV file is empty or has no header row");
                return;
            }
            int colCount = headerNames.size();
            ArrayList<ArrayList<String>> columns = new ArrayList<>();
            for (int i = 0; i < colCount; i++) {
                columns.add(new ArrayList<>());
            }

            for (CSVRecord record : parser) {
                for (int i = 0; i < colCount; i++) {
                    String v = i < record.size() ? record.get(i) : "";
                    columns.get(i).add(v != null ? v : "");
                }
            }

            for (int i = 0; i < colCount; i++) {
                ArrayList<String> col = columns.get(i);
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                java.io.DataOutputStream dos = new java.io.DataOutputStream(baos);
                ArrayList<Integer> intList = new ArrayList<>();
                ArrayList<Float> floatList = new ArrayList<>();

                if (tsType == TSDataType.INT32) {
                    for (String v : col) {
                        if (v == null) {
                            continue;
                        }
                        String t = v.trim();
                        if (t.isEmpty() || ExperimentDataPolicy.isNaNCell(t)) {
                            continue;
                        }
                        int intValue = Integer.parseInt(t);
                        dos.writeInt(intValue);
                        intList.add(intValue);
                    }
                } else {
                    for (String v : col) {
                        if (v == null) {
                            continue;
                        }
                        String t = v.trim();
                        if (t.isEmpty() || ExperimentDataPolicy.isNaNCell(t)) {
                            continue;
                        }
                        float floatValue = Float.parseFloat(t);
                        if (!ExperimentDataPolicy.keepNumericValue(floatValue)) continue;
                        dos.writeFloat(floatValue);
                        floatList.add(floatValue);
                    }
                }
                dos.close();

                if (baos.size() > 0) {
                    columnByteData.add(baos.toByteArray());
                    if (tsType == TSDataType.INT32) {
                        columnIntData.add(intList);
                    } else {
                        columnFloatData.add(floatList);
                    }
                }
            }
        }

        try (FileWriter fw = new FileWriter(appendPath, StandardCharsets.UTF_8, true);
                CSVPrinter csvWriter =
                        new CSVPrinter(fw, CSVFormat.DEFAULT.withRecordSeparator("\n"))) {

            if ("external-only".equals(phase)) {
                ExternalCompressionBench.printRows(
                        tsType,
                        BenchmarkColumnBridge.toLongColumns(columnIntData),
                        BenchmarkColumnBridge.toDoubleColumns(columnFloatData),
                        csvWriter);
            } else {
                System.err.println("Unknown --phase: " + phase);
                System.exit(2);
            }
            csvWriter.flush();
        }
    }
}
