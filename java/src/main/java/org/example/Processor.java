package org.example;

import org.apache.commons.compress.compressors.CompressorStreamFactory;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.CSVPrinter;
import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.CompressionType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.TsFileReader;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.query.dataset.QueryDataSet;
import org.apache.tsfile.write.TsFileWriter;
import org.apache.tsfile.write.record.TSRecord;
import org.apache.tsfile.write.record.datapoint.DoubleDataPoint;
import org.apache.tsfile.write.record.datapoint.FloatDataPoint;
import org.apache.tsfile.write.record.datapoint.IntDataPoint;
import org.apache.tsfile.write.record.datapoint.LongDataPoint;
import org.apache.tsfile.write.record.datapoint.StringDataPoint;
import org.apache.tsfile.write.schema.IMeasurementSchema;
import org.apache.tsfile.write.schema.MeasurementSchema;
import org.apache.tsfile.utils.Binary;

import java.io.*;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Processor {
    private static final List<String> NATIVE_REPLACEMENT_ROWS =
            List.of(
                    "blosc_blosclz",
                    "blosc2_blosclz",
                    "fpzip",
                    "fpcompress_dpratio",
                    "fpcompress_dpspeed",
                    "elf_on_chimp",
                    "elf_plus",
                    "camel");

    private static void printNativeReplacementRows(
            List<byte[]> columnByteData, CSVPrinter csvWriter, Set<String> existingAlgorithms)
            throws IOException {
        if (columnByteData == null || columnByteData.isEmpty()) {
            return;
        }
        for (String algorithm : NATIVE_REPLACEMENT_ROWS) {
            if (AlgorithmAllowList.shouldSkip(existingAlgorithms, algorithm)) {
                continue;
            }
            long[] result = new long[4];
            double[] lossAcc = new double[2];
            NativeLzFamilyBench.benchByteColumns(columnByteData, algorithm, result, lossAcc);
            if (result[0] > 0) {
                csvWriter.printRecord(algorithm, result[0], result[1], result[2], result[3]);
                existingAlgorithms.add(algorithm);
            }
        }
    }

    /** Column payloads: little-endian int64/float64 (matches C++/Python). */
    static void appendColumnLongLe(ByteArrayOutputStream baos, long value, ByteBuffer le) {
        BenchmarkWireFormat.appendColumnLongLe(baos, value, le);
    }

    static void appendColumnDoubleLe(ByteArrayOutputStream baos, double value, ByteBuffer le) {
        BenchmarkWireFormat.appendColumnDoubleLe(baos, value, le);
    }

    private static void appendColumnIntLe(ByteArrayOutputStream baos, int value, ByteBuffer le) {
        appendColumnLongLe(baos, value, le);
    }

    private static void appendColumnFloatLe(ByteArrayOutputStream baos, float value, ByteBuffer le) {
        appendColumnDoubleLe(baos, value, le);
    }

    /**
     * Write benchmark CSV to a temp file in the destination directory, then atomically replace the final path
     * so interrupts never leave a half-written CSV (e.g. truncated {@code sz3_} lines).
     */
    static File createAtomicReportTempFile(String outputReportPath) throws IOException {
        File dest = new File(outputReportPath);
        File dir = dest.getParentFile();
        if (dir != null && !dir.exists() && !dir.mkdirs()) {
            throw new IOException("Cannot create directory: " + dir);
        }
        return File.createTempFile(dest.getName() + ".", ".writing", dir);
    }

    static void replaceReportFileAtomically(File tmp, String outputReportPath) throws IOException {
        java.nio.file.Path t = tmp.toPath();
        java.nio.file.Path d = java.nio.file.Path.of(outputReportPath);
        try {
            Files.move(t, d, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(t, d, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Trim a CSV field and strip one pair of surrounding ASCII double quotes so a line
     * that is literally {@code ""} becomes empty and can be skipped (naive {@code split(",")} leaves
     * the two-quote token, which is not {@link String#isEmpty()}).
     */
    static String normalizeCsvNumericCell(String v) {
        if (v == null) {
            return "";
        }
        String t = v.trim();
        if (t.length() >= 2 && t.charAt(0) == '"' && t.charAt(t.length() - 1) == '"') {
            t = t.substring(1, t.length() - 1).trim();
        }
        return ExperimentDataPolicy.isNaNCell(t) ? "" : t;
    }

    static int fractionalDecimalPlacesFromCell(String numericCell) {
        if (numericCell == null || numericCell.isEmpty()) {
            return 0;
        }
        try {
            BigDecimal bd = new BigDecimal(numericCell).stripTrailingZeros();
            return Math.max(0, bd.scale());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    public static void processCsv(
            String csvPath,
            String dataType,
            int lzmaLevel,
            int zstdLevel,
            String outputReportPath) {
        File tmpReport = null;
        boolean reportReplaced = false;
        try {
            tmpReport = createAtomicReportTempFile(outputReportPath);
            try (
                    BufferedReader br = new BufferedReader(new FileReader(csvPath));
                    CSVPrinter csvWriter = new CSVPrinter(
                            new FileWriter(tmpReport, StandardCharsets.UTF_8),
                            CSVFormat.DEFAULT
                                    .withHeader("algorithm", "originalSize", "compressedSize",
                                            "compressionTime", "decompressionTime")
                                    .withRecordSeparator("\n"))) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                System.err.println("CSV file is empty");
                return;
            }

            String[] headers = headerLine.split(",");
            int colCount = headers.length;
            ArrayList<ArrayList<String>> columns = new ArrayList<>();
            for (int i = 0; i < colCount; i++) {
                columns.add(new ArrayList<>());
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                for (int i = 0; i < colCount && i < values.length; i++) {
                    columns.get(i).add(values[i]);
                }
            }

            TSDataType tsType;
            if ("int".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.INT32;
            } else if ("float".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.FLOAT;
            } else {
                // System.err.println("Only int32 and float types are supported");
                return;
            }

            ArrayList<Integer> validColumnIndexes = new ArrayList<>();
            ArrayList<String> validColumnHeaders = new ArrayList<>();
            ArrayList<byte[]> validColumnByteData = new ArrayList<>();
            ArrayList<ArrayList<Integer>> validColumnIntData = new ArrayList<>();
            ArrayList<ArrayList<Float>> validColumnFloatData = new ArrayList<>();

            for (int i = 0; i < colCount; i++) {
                ArrayList<String> col = columns.get(i);
                boolean valid = true;

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ByteBuffer le = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);

                ArrayList<Integer> intList = new ArrayList<>();
                ArrayList<Float> floatList = new ArrayList<>();

                if (tsType == TSDataType.INT32) {
                    for (String v : col) {
                        String t = normalizeCsvNumericCell(v);
                        if (t.isEmpty()) {
                            continue;
                        }
                        try {
                            int intValue = Integer.parseInt(t);
                            appendColumnIntLe(baos, intValue, le);
                            intList.add(intValue);
                        } catch (NumberFormatException e) {
                            valid = false;
                            break;
                        }
                    }
                } else if (tsType == TSDataType.FLOAT) {
                    for (String v : col) {
                        String t = normalizeCsvNumericCell(v);
                        if (t.isEmpty()) {
                            continue;
                        }
                        try {
                            float floatValue = Float.parseFloat(t);
                            if (!ExperimentDataPolicy.keepNumericValue(floatValue)) continue;
                            appendColumnFloatLe(baos, floatValue, le);
                            floatList.add(floatValue);
                        } catch (NumberFormatException e) {
                            valid = false;
                            break;
                        }
                    }
                }

                if (valid && baos.size() > 0) {

                    validColumnIndexes.add(i);
                    validColumnHeaders.add(headers[i]);
                    validColumnByteData.add(baos.toByteArray());

                    if (tsType == TSDataType.INT32) {
                        validColumnIntData.add(intList);
                    } else if (tsType == TSDataType.FLOAT) {
                        validColumnFloatData.add(floatList);
                    }
                }
            }

            List<String> algos = new ArrayList<>();
            algos.add(CompressorStreamFactory.GZIP);
            algos.add(CompressorStreamFactory.SNAPPY_FRAMED);
            algos.add(CompressorStreamFactory.LZ4_FRAMED);

            for (String algo : algos) {
                System.out.println("Processing compression algorithm: " + algo);
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testCommonCompression(data, algo, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord(algo, totalOriginal, totalCompressed, totalCompTime, totalDecompTime);
            }

            List<TSEncoding> encodingList = new ArrayList<>();
            encodingList.add(TSEncoding.TS_2DIFF);
            encodingList.add(TSEncoding.RLE);
            encodingList.add(TSEncoding.GORILLA);
            encodingList.add(TSEncoding.CHIMP);
            encodingList.add(TSEncoding.SPRINTZ);
            // encodingList.add(TSEncoding.ZIGZAG);
            encodingList.add(TSEncoding.RLBE);

            for (TSEncoding encoding : encodingList) {
                System.out.println("Processing encoding: " + encoding);
                long totalOriginal = 0;
                long totalEncoded = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    long[] result = new long[4];
                    try {
                        if (tsType == TSDataType.INT32) {
                            ArrayList<Integer> intList = validColumnIntData.get(idx);
                            CompressionUtils.testEncoding(tsType, encoding, intList, result);
                        } else if (tsType == TSDataType.FLOAT) {
                            ArrayList<Float> floatList = validColumnFloatData.get(idx);
                            CompressionUtils.testEncoding(tsType, encoding, floatList, result);
                        } else {
                            throw new IllegalArgumentException("Unsupported data type: " + tsType);
                        }
                        totalOriginal += result[0];
                        totalEncoded += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord(encoding, totalOriginal, totalEncoded, totalCompTime, totalDecompTime);
            }

            printColumnarEncodingRows(
                    tsType,
                    validColumnIntData,
                    validColumnFloatData,
                    csvWriter,
                    new LinkedHashSet<>());
            Set<String> nativeReplacementExistingAlgorithms = new LinkedHashSet<>();
            printNativeReplacementRows(validColumnByteData, csvWriter, nativeReplacementExistingAlgorithms);
            System.setProperty(
                    AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY,
                    AlgorithmAllowList.toEnvValue(nativeReplacementExistingAlgorithms));
            ExternalCompressionBench.printRows(
                    tsType,
                    BenchmarkColumnBridge.toLongColumns(validColumnIntData),
                    BenchmarkColumnBridge.toDoubleColumns(validColumnFloatData),
                    csvWriter);
            ResearchNativeCliBench.printRows(tsType, validColumnIntData, validColumnFloatData, csvWriter);
            ResearchFpzipCliBench.printRows(tsType, validColumnIntData, validColumnFloatData, csvWriter);
            ResearchBloscCliBench.printRows64(
                    tsType,
                    BenchmarkColumnBridge.toLongColumns(validColumnIntData),
                    BenchmarkColumnBridge.toDoubleColumns(validColumnFloatData),
                    csvWriter);
            ResearchBlosc2CliBench.printRows64(
                    tsType,
                    BenchmarkColumnBridge.toLongColumns(validColumnIntData),
                    BenchmarkColumnBridge.toDoubleColumns(validColumnFloatData),
                    csvWriter);
            ResearchFpcompressCliBench.printRows64(
                    tsType,
                    BenchmarkColumnBridge.toLongColumns(validColumnIntData),
                    BenchmarkColumnBridge.toDoubleColumns(validColumnFloatData),
                    csvWriter);
                ResearchSerfCliBench.printRows(tsType, validColumnIntData, validColumnFloatData, csvWriter);
                ResearchBuffCliBench.printRows(tsType, validColumnIntData, validColumnFloatData, csvWriter);
                ResearchCmixCliBench.printRows(tsType, validColumnIntData, validColumnFloatData, csvWriter);
                ResearchZfpCliBench.printRows64(
                    tsType,
                    BenchmarkColumnBridge.toLongColumns(validColumnIntData),
                    BenchmarkColumnBridge.toDoubleColumns(validColumnFloatData),
                    csvWriter);

            // process XZ/LZMA2
            System.out.println("Processing XZ compression");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testLZMACompression(data, lzmaLevel, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("xz", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

            // brotli
            System.out.println("Processing Brotli compression");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testBrotliCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("brotli", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

            // process zstd
            System.out.println("Processing Zstandard compression");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testZstdCompression(data, zstdLevel, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("zstd", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

            // ans (order-0 rANS entropy coding)
            System.out.println("Processing ANS (rANS) entropy coding");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testAnsEntropyCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("ans", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

            // huffman (byte Huffman coding)
            System.out.println("Processing Huffman (byte) coding");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testHuffmanCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("huffman", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

            // arithmetic (static order-0 range coding)
            System.out.println("Processing Arithmetic (byte) coding");
            {
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int idx = 0; idx < validColumnIndexes.size(); idx++) {
                    int i = validColumnIndexes.get(idx);
                    System.out.println("Processing column " + i + " (" + validColumnHeaders.get(idx) + ")");
                    byte[] data = validColumnByteData.get(idx);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testArithmeticCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("arithmetic", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
            }

                csvWriter.flush();
            }
            replaceReportFileAtomically(tmpReport, outputReportPath);
            reportReplaced = true;
            System.out.println("Compression results written to " + outputReportPath);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (!reportReplaced && tmpReport != null) {
                try {
                    Files.deleteIfExists(tmpReport.toPath());
                } catch (IOException ignored) {
                }
            }
        }
    }

    public static void processCsvSimple(
            String csvPath,
            String dataType,
            String outputReportPath) {
        processCsvSimple(csvPath, dataType, outputReportPath, null, null);
    }

    /** Home / catalog staticdata: optional {@code {dataset}_column_info.csv}; reads CSV in row patches. */
    public static void processCsvSimple(
            String csvPath,
            String dataType,
            String outputReportPath,
            String columnInfoPath) {
        processCsvSimple(csvPath, dataType, outputReportPath, columnInfoPath, null);
    }

    public static void processCsvSimple(
            String csvPath,
            String dataType,
            String outputReportPath,
            String columnInfoPath,
            Integer columnIndex) {
        ProcessorPatchBench.run(csvPath, dataType, outputReportPath, columnInfoPath, columnIndex);
    }

    @SuppressWarnings("unused")
    private static void processCsvSimpleLoadAllLegacy(
            String csvPath,
            String dataType,
            String outputReportPath) {
        try {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_STATICDATA_REGEN"))
                    && new File(outputReportPath).isFile()) {
                if (!new File(outputReportPath).delete()) {
                    System.err.println("Warning: could not delete " + outputReportPath + " for regen");
                }
            }
            boolean appendMode = new File(outputReportPath).isFile();
            // Honor COMPRESSION_BENCHMARK_SKIP_ALGORITHMS (e.g. Python java supplement) before incremental CSV rows.
            Set<String> existingAlgorithms = new LinkedHashSet<>(AlgorithmAllowList.fromIncrementalSkipEnv());
            if (!AlgorithmAllowList.hasAllowlist()) {
                existingAlgorithms.addAll(
                        AlgorithmAllowList.fromCsvReport(java.nio.file.Path.of(outputReportPath)));
            }
            try (
                BufferedReader br = new BufferedReader(new FileReader(csvPath));
                CSVPrinter csvWriter = new CSVPrinter(
                        new FileWriter(outputReportPath, StandardCharsets.UTF_8, appendMode),
                        CSVFormat.DEFAULT.withRecordSeparator("\n"))) {
            if (!appendMode) {
                csvWriter.printRecord("algorithm", "originalSize", "compressedSize",
                        "compressionTime", "decompressionTime");
            }
            String headerLine = br.readLine();
            if (headerLine == null) {
                System.err.println("CSV file is empty");
                return;
            }

            String[] headers = headerLine.split(",");
            int colCount = headers.length;
            ArrayList<ArrayList<String>> columns = new ArrayList<>();
            for (int i = 0; i < colCount; i++) {
                columns.add(new ArrayList<>());
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                for (int i = 0; i < colCount && i < values.length; i++) {
                    columns.get(i).add(values[i]);
                }
            }

            TSDataType tsType;
            if ("int".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.INT64;
            } else if ("float".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.DOUBLE;
            } else {
                return;
            }

            ArrayList<byte[]> columnByteData = new ArrayList<>();
            ArrayList<ArrayList<Long>> columnIntData = new ArrayList<>();
            ArrayList<ArrayList<Double>> columnFloatData = new ArrayList<>();
            ArrayList<Integer> columnFloatMaxPointData = new ArrayList<>();

            for (int i = 0; i < colCount; i++) {
                ArrayList<String> col = columns.get(i);

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ByteBuffer le = ByteBuffer.allocate(BenchmarkWireFormat.INT_BYTES).order(ByteOrder.LITTLE_ENDIAN);

                ArrayList<Long> intList = new ArrayList<>();
                ArrayList<Double> floatList = new ArrayList<>();
                int floatMaxPoint = 0;

                if (tsType == TSDataType.INT64) {
                    for (String v : col) {
                        String t = normalizeCsvNumericCell(v);
                        if (!t.isEmpty()) {
                            long intValue = Long.parseLong(t);
                            appendColumnLongLe(baos, intValue, le);
                            intList.add(intValue);
                        }
                    }
                } else if (tsType == TSDataType.DOUBLE) {
                    for (String v : col) {
                        String t = normalizeCsvNumericCell(v);
                        if (!t.isEmpty()) {
                            try {
                                double floatValue = Double.parseDouble(t);
                                if (!ExperimentDataPolicy.keepNumericValue(floatValue)) continue;
                                appendColumnDoubleLe(baos, floatValue, le);
                                floatList.add(floatValue);
                                floatMaxPoint = Math.max(floatMaxPoint, fractionalDecimalPlacesFromCell(t));
                            } catch (NumberFormatException ignored) {
                                // skip non-numeric cells (match C++/Python)
                            }
                        }
                    }
                }

                if (baos.size() > 0) {
                    columnByteData.add(baos.toByteArray());
                    if (tsType == TSDataType.INT64) {
                        columnIntData.add(intList);
                    } else if (tsType == TSDataType.DOUBLE) {
                        columnFloatData.add(floatList);
                        columnFloatMaxPointData.add(floatMaxPoint);
                    }
                }
            }

            List<String> algos = new ArrayList<>();
            algos.add(CompressorStreamFactory.GZIP);
            algos.add(CompressorStreamFactory.SNAPPY_FRAMED);
            algos.add(CompressorStreamFactory.LZ4_FRAMED);

            for (String algo : algos) {
                if (AlgorithmAllowList.shouldSkip(existingAlgorithms, algo)) {
                    System.out.println("Skip existing algorithm: " + algo);
                    continue;
                }
                System.out.println("Processing compression algorithm: " + algo);
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testCommonCompression(data, algo, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord(algo, totalOriginal, totalCompressed, totalCompTime, totalDecompTime);
                existingAlgorithms.add(algo);
            }

            List<TSEncoding> encodingList = new ArrayList<>();
            encodingList.add(TSEncoding.TS_2DIFF);
            encodingList.add(TSEncoding.RLE);
            encodingList.add(TSEncoding.GORILLA);
            encodingList.add(TSEncoding.CHIMP);
            encodingList.add(TSEncoding.SPRINTZ);
            encodingList.add(TSEncoding.RLBE);

            for (TSEncoding encoding : encodingList) {
                String rowName = encoding.toString();
                if (AlgorithmAllowList.shouldSkip(existingAlgorithms, rowName)) {
                    System.out.println("Skip existing algorithm: " + rowName);
                    continue;
                }
                System.out.println("Processing encoding: " + encoding);
                long totalOriginal = 0;
                long totalEncoded = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    long[] result = new long[4];
                    try {
                        if (tsType == TSDataType.INT64) {
                            ArrayList<Long> intList = columnIntData.get(i);
                            CompressionUtils.testEncoding(tsType, encoding, intList, result);
                        } else if (tsType == TSDataType.DOUBLE) {
                            ArrayList<Double> floatList = columnFloatData.get(i);
                            int maxPoint = i < columnFloatMaxPointData.size() ? columnFloatMaxPointData.get(i) : 0;
                            CompressionUtils.testEncoding(tsType, encoding, floatList, maxPoint, result);
                        } else {
                            throw new IllegalArgumentException("Unsupported data type: " + tsType);
                        }
                        totalOriginal += result[0];
                        totalEncoded += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord(encoding, totalOriginal, totalEncoded, totalCompTime, totalDecompTime);
                existingAlgorithms.add(rowName);
            }

            System.out.println(
                    "Benchmark phase: columnar encodings (dictionary / BOS / …) — no per-column log here");
            printColumnarEncodingRows64(
                    tsType, columnIntData, columnFloatData, columnFloatMaxPointData, csvWriter, existingAlgorithms);
            System.out.println("Benchmark phase: native replacement byte codecs (Blosc/FPZIP/FPCompress/ELF/Camel)");
            printNativeReplacementRows(columnByteData, csvWriter, existingAlgorithms);
            System.out.println(
                    "Benchmark phase: ExternalCompressionBench (FastPFOR, Chimp, SPDP, Elf, …) — can be slow");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ExternalCompressionBench.printRows(tsType, columnIntData, columnFloatData, csvWriter);
            ArrayList<ArrayList<Integer>> legacyInt = BenchmarkColumnBridge.toIntegerColumns(columnIntData);
            ArrayList<ArrayList<Float>> legacyFloat = BenchmarkColumnBridge.toFloatColumns(columnFloatData);
            System.out.println(
                    "Benchmark phase: ResearchNativeCliBench (SZ/SZ3; COMPRESSION_BENCHMARK_SZ_BENCH / SZ3_BENCH)");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchNativeCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println("Benchmark phase: fpzip native replacement already emitted; legacy CLI skipped");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchFpzipCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println("Benchmark phase: Blosc native replacement already emitted; legacy CLI skipped");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchBloscCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println("Benchmark phase: Blosc2 native replacement already emitted; legacy CLI skipped");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchBlosc2CliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println("Benchmark phase: FPCompress native replacement already emitted; legacy CLI skipped");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchFpcompressCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println(
                    "Benchmark phase: ResearchSerfCliBench (skipped unless COMPRESSION_BENCHMARK_SERF_BENCH=1)");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchSerfCliBench.printRows(tsType, legacyInt, legacyFloat, csvWriter);
            System.out.println("Benchmark phase: ResearchBuffCliBench (BUFF / kernel)");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchBuffCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);
            System.out.println(
                    "Benchmark phase: ResearchCmixCliBench (off unless COMPRESSION_BENCHMARK_CMIX_BENCH=1)");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchCmixCliBench.printRows(tsType, legacyInt, legacyFloat, csvWriter);
            System.out.println(
                    "Benchmark phase: ResearchZfpCliBench (LLNL zfp CLI; COMPRESSION_BENCHMARK_ZFP_BENCH)");
            System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(existingAlgorithms));
            ResearchZfpCliBench.printRows64(tsType, columnIntData, columnFloatData, csvWriter);

            // process XZ/LZMA2 with levels 0-9
            for (int level = 0; level <= 9; level++) {
                String rowName = "xz_level_" + level;
                if (AlgorithmAllowList.shouldSkip(existingAlgorithms, rowName)) {
                    System.out.println("Skip existing algorithm: " + rowName);
                    continue;
                }
                System.out.println("Processing XZ compression level " + level);
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testLZMACompression(data, level, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("xz_level_" + level, totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add(rowName);
            }

            // brotli
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "brotli")) {
                System.out.println("Processing brotli compression");
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testBrotliCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("brotli", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add("brotli");
            }

            // process zstd with levels 1-9
            for (int level = 1; level <= 9; level++) {
                String rowName = "zstd_level_" + level;
                if (AlgorithmAllowList.shouldSkip(existingAlgorithms, rowName)) {
                    System.out.println("Skip existing algorithm: " + rowName);
                    continue;
                }
                System.out.println("Processing zstd compression level " + level);
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testZstdCompression(data, level, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("zstd_level_" + level, totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add(rowName);
            }

            // ans (order-0 rANS entropy coding)
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "ans")) {
                System.out.println("Processing ANS (rANS) entropy coding");
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testAnsEntropyCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("ans", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add("ans");
            }

            // huffman (byte Huffman coding)
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "huffman")) {
                System.out.println("Processing Huffman (byte) coding");
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testHuffmanCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("huffman", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add("huffman");
            }

            // arithmetic (static order-0 range coding)
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "arithmetic")) {
                System.out.println("Processing Arithmetic (byte) coding");
                long totalOriginal = 0;
                long totalCompressed = 0;
                long totalCompTime = 0;
                long totalDecompTime = 0;

                for (int i = 0; i < columnByteData.size(); i++) {
                    System.out.println("Processing column " + i + " (" + headers[i] + ")");
                    byte[] data = columnByteData.get(i);
                    long[] result = new long[4];
                    try {
                        CompressionUtils.testArithmeticCompression(data, result);
                        totalOriginal += result[0];
                        totalCompressed += result[1];
                        totalCompTime += result[2];
                        totalDecompTime += result[3];
                    } catch (Exception e) {
                        continue;
                    }
                }

                csvWriter.printRecord("arithmetic", totalOriginal, totalCompressed, totalCompTime,
                        totalDecompTime);
                existingAlgorithms.add("arithmetic");
            }

                csvWriter.flush();
            }
            System.out.println("Compression results written to " + outputReportPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Per-column max fractional decimal digits for FLOAT measurements (from Python sidecar).
     * Returns {@code null} when no sidecar path is given (caller uses global fallback).
     */
    private static int[] readColumnMaxPointsOrNull(
            String columnMaxPointsPath, int colCount, int fallback) throws IOException {
        if (columnMaxPointsPath == null || columnMaxPointsPath.isBlank()) {
            return null;
        }
        java.nio.file.Path path = java.nio.file.Paths.get(columnMaxPointsPath);
        if (!Files.isRegularFile(path)) {
            throw new FileNotFoundException("Column max-points file not found: " + columnMaxPointsPath);
        }
        int[] out = new int[colCount];
        Arrays.fill(out, Math.max(0, fallback));
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line = br.readLine();
            if (line == null) {
                return out;
            }
            StringBuilder csvChunk = new StringBuilder();
            csvChunk.append(line).append('\n');
            while ((line = br.readLine()) != null) {
                csvChunk.append(line).append('\n');
            }
            try (Reader fr = new StringReader(csvChunk.toString());
                    CSVParser parser =
                            CSVFormat.DEFAULT
                                    .builder()
                                    .setHeader("column_index", "column_name", "max_point_number")
                                    .setSkipHeaderRecord(true)
                                    .setIgnoreEmptyLines(true)
                                    .setIgnoreHeaderCase(true)
                                    .setTrim(true)
                                    .build()
                                    .parse(fr)) {
                for (CSVRecord rec : parser) {
                    int idx = Integer.parseInt(rec.get("column_index"));
                    int mp = Integer.parseInt(rec.get("max_point_number"));
                    if (idx >= 0 && idx < colCount) {
                        out[idx] = Math.max(0, mp);
                    }
                }
            }
        }
        return out;
    }

    private static int maxPointForColumn(
            int colIndex,
            TSDataType tsType,
            int[] perColumn,
            int fallback,
            ArrayList<Integer> parsedFloatMaxByHeader) {
        if (tsType != TSDataType.DOUBLE) {
            return 0;
        }
        int fromParsed = 0;
        if (parsedFloatMaxByHeader != null
                && colIndex >= 0
                && colIndex < parsedFloatMaxByHeader.size()) {
            fromParsed = Math.max(0, parsedFloatMaxByHeader.get(colIndex));
        }
        if (perColumn != null && colIndex >= 0 && colIndex < perColumn.length) {
            int sidecar = Math.max(0, perColumn[colIndex]);
            if (sidecar > 0) {
                return Math.max(fromParsed, sidecar);
            }
        }
        if (fromParsed > 0) {
            return fromParsed;
        }
        return Math.max(0, fallback);
    }

    /** Display / CSV keys for TsFile chunk compressors (PLAIN encoding). */
    private static String tsfileChunkCompressionAlgorithmKey(CompressionType c) {
        switch (c) {
            case SNAPPY:
                return "snappy";
            case GZIP:
                return "gzip";
            case LZ4:
                return "lz4";
            case ZSTD:
                return "zstd";
            case LZMA2:
                return "XZ";
            default:
                return c.name();
        }
    }

    public static void processCsvSimple(
            String csvPath,
            String dataType,
            int maxPointNumberFallback,
            String columnMaxPointsPath,
            String tsfileDir,
            String outputReportPath) {
        processCsvSimple(
                csvPath,
                dataType,
                maxPointNumberFallback,
                null,
                columnMaxPointsPath,
                tsfileDir,
                outputReportPath);
    }

    /** TsFile staticdata: stream rows per algorithm; {@code columnInfoPath} preferred over legacy max-points sidecar. */
    public static void processCsvSimple(
            String csvPath,
            String dataType,
            int maxPointNumberFallback,
            String columnInfoPath,
            String columnMaxPointsPath,
            String tsfileDir,
            String outputReportPath) {
        ProcessorTsfileStaticdataBench.run(
                csvPath,
                dataType,
                maxPointNumberFallback,
                columnInfoPath,
                columnMaxPointsPath,
                tsfileDir,
                outputReportPath);
    }


    public static void processCsvToTsfile(
            String csvPath,
            String dataType,
            int maxPointNumber,
            String tsFileDir,
            String outputReportPath) {
        try {
            File outputDir = new File(tsFileDir);
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }

            String csvFileName = new File(csvPath).getName();
            String baseName = csvFileName.substring(0, csvFileName.lastIndexOf('.'));

            TSDataType tsType;
            if ("int".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.INT32;
            } else if ("float".equalsIgnoreCase(dataType)) {
                tsType = TSDataType.FLOAT;
            } else {
                return;
            }

            List<TSEncoding> encodingList = new ArrayList<>();
            encodingList.add(TSEncoding.PLAIN);
            encodingList.add(TSEncoding.TS_2DIFF);
            encodingList.add(TSEncoding.RLE);
            encodingList.add(TSEncoding.GORILLA);
            encodingList.add(TSEncoding.CHIMP);
            encodingList.add(TSEncoding.SPRINTZ);
            encodingList.add(TSEncoding.RLBE);

            List<CompressionType> compressionList = new ArrayList<>();
            compressionList.add(CompressionType.UNCOMPRESSED);
            compressionList.add(CompressionType.SNAPPY);
            compressionList.add(CompressionType.GZIP);
            compressionList.add(CompressionType.LZ4);
            compressionList.add(CompressionType.ZSTD);
            compressionList.add(CompressionType.LZMA2);

            try (CSVPrinter csvWriter = new CSVPrinter(
                    new FileWriter(outputReportPath),
                    CSVFormat.DEFAULT
                            .withHeader("filePath", "encodingMethod", "compressionMethod", "fileSize", "writeTime")
                            .withRecordSeparator("\n"))) {

                for (TSEncoding encoding : encodingList) {
                    for (CompressionType compression : compressionList) {
                        System.out.println("Processing encoding: " + encoding + ", compression: " + compression);

                        String tsFileName = baseName + "_" + encoding.toString() + "_" + compression.toString()
                                + ".tsfile";
                        String tsFilePath = new File(outputDir, tsFileName).getAbsolutePath();

                        try {
                            // CsvProcessor.csvToTsFile(csvPath, tsFilePath, tsType, encoding, compression,
                            // maxPointNumber);

                            long[] result = new long[1];

                            Processor.csvToTsFileSimple(csvPath, tsFilePath, tsType, encoding, compression,
                                    maxPointNumber, result);

                            File tsFile = new File(tsFilePath);
                            long fileSize = tsFile.exists() ? tsFile.length() : 0;

                            csvWriter.printRecord(tsFilePath, encoding.toString(), compression.toString(), fileSize,
                                    result[0]);

                            System.out.println(
                                    "Successfully generated: " + tsFileName + " (size: " + fileSize + " bytes)");
                            System.out.println("Write time (nanoseconds): " + result[0]);
                        } catch (Exception e) {
                            System.err.println(
                                    "Error processing " + encoding + "_" + compression + ": " + e.getMessage());
                            e.printStackTrace();
                        }
                    }
                }

                csvWriter.flush();
                System.out.println("TsFile information written to: " + outputReportPath);
            }

            System.out.println("All TSFiles generated successfully, output directory: " + tsFileDir);

        } catch (Exception e) {
            System.err.println("Error processing CSV to TSFile: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void csvToTsFile(
            String csvPath,
            String tsfilePath,
            TSDataType dataType,
            TSEncoding encoding,
            CompressionType compressionType,
            int maxPointNumber) throws Exception {
        File file = new File(tsfilePath);

        try (TsFileWriter writer = new TsFileWriter(file);
                BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String headerLine = br.readLine();

            if (headerLine == null) {
                System.out.println("CSV file is empty");
                return;
            }

            String[] headers = headerLine.split(",");

            List<String[]> allRows = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                allRows.add(line.split(",", -1));
            }

            if (allRows.isEmpty()) {
                System.out.println("Warning: CSV file contains only headers, no data rows.");
                return;
            }

            String deviceId = "device_1";
            Path devicePath = new Path(deviceId);

            List<IMeasurementSchema> measurementSchemas = new ArrayList<>();
            List<Integer> validColumnIndices = new ArrayList<>();

            // Validate all columns, add those that can be parsed as the specified data type
            // to validColumnIndices
            for (int i = 0; i < headers.length; i++) {
                boolean isColumnValid = true;
                for (String[] row : allRows) {
                    if (i < row.length && row[i] != null) {
                        String t = normalizeCsvNumericCell(row[i]);
                        if (t.isEmpty()) {
                            continue;
                        }
                        try {
                            if (dataType == TSDataType.INT32) {
                                Integer.parseInt(t);
                            } else if (dataType == TSDataType.FLOAT) {
                                Float.parseFloat(t);
                            }
                        } catch (NumberFormatException e) {
                            isColumnValid = false;
                            System.out.println("Column '" + headers[i]
                                    + "' ignored due to values that cannot be parsed as " + dataType + ".");
                            break;
                        }
                    }
                }

                if (isColumnValid) {
                    String measurementName = headers[i];

                    Map<String, String> props = new HashMap<>();
                    props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(maxPointNumber));

                    MeasurementSchema schema = new MeasurementSchema(
                            measurementName,
                            dataType,
                            encoding,
                            compressionType,
                            props);

                    measurementSchemas.add(schema);
                    validColumnIndices.add(i);
                }
            }

            if (measurementSchemas.isEmpty()) {
                System.err.println("No valid data columns found that can be successfully parsed as " + dataType + ".");
                return;
            }

            writer.registerTimeseries(devicePath, measurementSchemas);

            for (int rowIndex = 0; rowIndex < allRows.size(); rowIndex++) {
                String[] values = allRows.get(rowIndex);

                long timestamp = rowIndex;
                TSRecord record = new TSRecord(deviceId, timestamp);

                for (int colIndex : validColumnIndices) {
                    if (colIndex < values.length && values[colIndex] != null) {
                        String t = normalizeCsvNumericCell(values[colIndex]);
                        if (t.isEmpty()) {
                            continue;
                        }
                        String measurementId = headers[colIndex];
                        if (dataType == TSDataType.INT32) {
                            record.addTuple(new IntDataPoint(measurementId, Integer.parseInt(t)));
                        } else if (dataType == TSDataType.FLOAT) {
                            float value = Float.parseFloat(t);
                            if (!ExperimentDataPolicy.keepNumericValue(value)) continue;
                            record.addTuple(new FloatDataPoint(measurementId, value));
                        }
                    }
                }
                writer.writeRecord(record);
            }
        }
    }

    public static void csvToTsFileSimple(
            String csvPath,
            String tsfilePath,
            TSDataType dataType,
            TSEncoding encoding,
            CompressionType compressionType,
            int maxPointNumber,
            long[] result) throws Exception {
        File file = new File(tsfilePath);

        try (TsFileWriter writer = new TsFileWriter(file);
                BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String headerLine = br.readLine();

            if (headerLine == null) {
                System.out.println("CSV file is empty");
                return;
            }

            String[] headers = headerLine.split(",");

            List<String[]> allRows = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                allRows.add(line.split(",", -1));
            }

            if (allRows.isEmpty()) {
                System.out.println("Warning: CSV file contains only headers, no data rows.");
                return;
            }

            long startWrite = System.nanoTime();

            String deviceId = "device_1";
            Path devicePath = new Path(deviceId);

            List<IMeasurementSchema> measurementSchemas = new ArrayList<>();

            for (int i = 0; i < headers.length; i++) {
                String measurementName = headers[i];

                Map<String, String> props = new HashMap<>();
                props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(maxPointNumber));

                MeasurementSchema schema = new MeasurementSchema(
                        measurementName,
                        dataType,
                        encoding,
                        compressionType,
                        props);

                measurementSchemas.add(schema);
            }

            writer.registerTimeseries(devicePath, measurementSchemas);

            for (int rowIndex = 0; rowIndex < allRows.size(); rowIndex++) {
                String[] values = allRows.get(rowIndex);

                long timestamp = rowIndex;
                TSRecord record = new TSRecord(deviceId, timestamp);

                // 直接处理所有列
                for (int colIndex = 0; colIndex < headers.length; colIndex++) {
                    if (colIndex < values.length && values[colIndex] != null) {
                        String t = normalizeCsvNumericCell(values[colIndex]);
                        if (t.isEmpty()) {
                            continue;
                        }
                        String measurementId = headers[colIndex];
                        if (dataType == TSDataType.INT32) {
                            record.addTuple(new IntDataPoint(measurementId, Integer.parseInt(t)));
                        } else if (dataType == TSDataType.FLOAT) {
                            float value = Float.parseFloat(t);
                            if (!ExperimentDataPolicy.keepNumericValue(value)) continue;
                            record.addTuple(new FloatDataPoint(measurementId, value));
                        }
                    }
                }
                writer.writeRecord(record);
            }

            long endWrite = System.nanoTime();
            long writeTime = endWrite - startWrite;
            result[0] = writeTime;
        }
    }

    private static final class ColumnLogicalType {
        final int index;
        final String name;
        final String logicalType; // integer | float | string
        final int maxPointNumber;

        ColumnLogicalType(int index, String name, String logicalType, int maxPointNumber) {
            this.index = index;
            this.name = name;
            this.logicalType = logicalType;
            this.maxPointNumber = Math.max(0, maxPointNumber);
        }
    }

    private static final class TypedExportManifest {
        final int maxPointNumber;
        final List<ColumnLogicalType> columns;

        TypedExportManifest(int maxPointNumber, List<ColumnLogicalType> columns) {
            this.maxPointNumber = maxPointNumber;
            this.columns = columns;
        }
    }

    /**
     * Reads {@code {key}_column_info.csv} (column_index,column_name,logical_type,max_point_number) or legacy
     * {@code column_types.csv} with {@code # max_point_number=N}.
     */
    private static TypedExportManifest readTypedExportManifest(String manifestPath) throws IOException {
        java.nio.file.Path manifestFile = java.nio.file.Paths.get(manifestPath);
        if (!Files.isRegularFile(manifestFile)) {
            throw new FileNotFoundException("Manifest not found: " + manifestPath);
        }
        try (BufferedReader br = Files.newBufferedReader(manifestFile, StandardCharsets.UTF_8)) {
            String header = br.readLine();
            if (header == null) {
                throw new IOException("Manifest is empty: " + manifestPath);
            }
            if (header.toLowerCase().contains("max_point_number")
                    && header.toLowerCase().contains("logical_type")) {
                return readTypedExportManifestFromColumnInfoHeader(header, br);
            }
        }
        return readTypedExportManifestLegacy(manifestPath);
    }

    private static TypedExportManifest readTypedExportManifestFromColumnInfoHeader(
            String headerLine, BufferedReader br) throws IOException {
        List<ColumnLogicalType> out = new ArrayList<>();
        int maxPointNumber = 0;
        StringBuilder csvChunk = new StringBuilder();
        csvChunk.append(headerLine).append('\n');
        String line;
        while ((line = br.readLine()) != null) {
            csvChunk.append(line).append('\n');
        }
        try (Reader fr = new StringReader(csvChunk.toString());
                CSVParser parser =
                        CSVFormat.DEFAULT
                                .builder()
                                .setHeader(
                                        "column_index",
                                        "column_name",
                                        "logical_type",
                                        "max_point_number")
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(true)
                                .setIgnoreHeaderCase(true)
                                .setTrim(true)
                                .build()
                                .parse(fr)) {
            for (CSVRecord rec : parser) {
                int idx = Integer.parseInt(rec.get("column_index"));
                String name = rec.get("column_name");
                String logical = rec.get("logical_type").toLowerCase();
                int mp = 0;
                try {
                    mp = Integer.parseInt(rec.get("max_point_number"));
                } catch (NumberFormatException ignored) {
                    // 0
                }
                if ("float".equals(logical)) {
                    maxPointNumber = Math.max(maxPointNumber, Math.max(0, mp));
                }
                if (!logical.equals("integer") && !logical.equals("float") && !logical.equals("string")) {
                    throw new IOException("Unsupported logical_type in column_info: " + logical);
                }
                out.add(new ColumnLogicalType(idx, name, logical, mp));
            }
        }
        out.sort((a, b) -> Integer.compare(a.index, b.index));
        return new TypedExportManifest(maxPointNumber, out);
    }

    private static TypedExportManifest readTypedExportManifestLegacy(String manifestPath) throws IOException {
        List<String> bodyLines = new ArrayList<>();
        int maxPointNumber = -1;
        try (BufferedReader br = Files.newBufferedReader(java.nio.file.Path.of(manifestPath), StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("#")) {
                    String rest = trimmed.substring(1).trim();
                    if (rest.startsWith("max_point_number=")) {
                        String num = rest.substring("max_point_number=".length()).trim();
                        try {
                            maxPointNumber = Integer.parseInt(num);
                        } catch (NumberFormatException e) {
                            throw new IOException("Invalid max_point_number in manifest: " + rest, e);
                        }
                    }
                    continue;
                }
                if (trimmed.isEmpty()) {
                    continue;
                }
                bodyLines.add(line);
            }
        }
        if (maxPointNumber < 0) {
            throw new IOException(
                    "Manifest must contain '# max_point_number=<non-negative int>' before the CSV header: "
                            + manifestPath);
        }
        List<ColumnLogicalType> out = new ArrayList<>();
        StringBuilder csvChunk = new StringBuilder();
        for (String l : bodyLines) {
            csvChunk.append(l).append('\n');
        }
        try (Reader fr = new StringReader(csvChunk.toString());
                CSVParser parser =
                        CSVFormat.DEFAULT
                                .builder()
                                .setHeader("column_index", "column_name", "logical_type")
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(true)
                                .setIgnoreHeaderCase(true)
                                .setTrim(true)
                                .build()
                                .parse(fr)) {
            for (CSVRecord rec : parser) {
                int idx = Integer.parseInt(rec.get("column_index"));
                String name = rec.get("column_name");
                String logical = rec.get("logical_type").toLowerCase();
                if (!logical.equals("integer") && !logical.equals("float") && !logical.equals("string")) {
                    throw new IOException("Unsupported logical_type in manifest: " + logical);
                }
                int mp = "float".equals(logical) ? maxPointNumber : 0;
                out.add(new ColumnLogicalType(idx, name, logical, mp));
            }
        }
        out.sort((a, b) -> Integer.compare(a.index, b.index));
        return new TypedExportManifest(maxPointNumber, out);
    }

    private static String measurementIdForColumn(int index) {
        return "col_" + index;
    }

    private static List<TSEncoding> supportedEncodingsFor(TSDataType tsType) {
        List<TSEncoding> encs = new ArrayList<>();
        for (TSEncoding enc : TSEncoding.values()) {
            if (TSEncoding.isSupported(tsType, enc)) {
                encs.add(enc);
            }
        }
        if (encs.isEmpty()) {
            encs.add(TSEncoding.PLAIN);
        }
        return encs;
    }

    private static TSEncoding parseOptionalTsEncoding(String name, String label) throws IOException {
        if (name == null || name.isBlank() || "NONE".equalsIgnoreCase(name.trim())) {
            return null;
        }
        try {
            return TSEncoding.valueOf(name.trim());
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid " + label + " encoding: " + name, e);
        }
    }

    /**
     * Export one catalog TsFile using the table model (TsFile ≥ 2.3.x) with fixed encodings and LZ4.
     * Legacy int/float/string encoding parameters are ignored (kept for script compatibility).
     */
    public static void exportTypedCsvToSelectedTsFile(
            String csvPath,
            String columnTypesManifestPath,
            String outputTsFilePath,
            String infoCsvPath,
            String intEncodingName,
            String floatEncodingName,
            String stringEncodingName)
            throws Exception {
        String tableName = new File(outputTsFilePath).getName();
        int dot = tableName.lastIndexOf('.');
        if (dot > 0) {
            tableName = tableName.substring(0, dot);
        }
        CatalogTableModelTsFileExport.export(
                csvPath, columnTypesManifestPath, outputTsFilePath, tableName);

        File tsFile = new File(outputTsFilePath);
        long size = Files.exists(tsFile.toPath()) ? Files.size(tsFile.toPath()) : 0L;
        java.nio.file.Path infoPath = java.nio.file.Paths.get(infoCsvPath);
        if (infoPath.getParent() != null) {
            Files.createDirectories(infoPath.getParent());
        }
        String resolvedTable = CatalogTableModelTsFileExport.sanitizeTableNameForInfo(tableName);
        try (CSVPrinter summary =
                new CSVPrinter(
                        Files.newBufferedWriter(infoPath, StandardCharsets.UTF_8),
                        CSVFormat.DEFAULT
                                .withHeader(
                                        "tsfile_model",
                                        "table_name",
                                        "tag_column_count",
                                        "int_encoding",
                                        "float_encoding",
                                        "string_encoding",
                                        "time_encoding",
                                        "compression",
                                        "file_size_bytes")
                                .withRecordSeparator("\n"))) {
            summary.printRecord(
                    CatalogTableModelTsFileExport.TSFILE_MODEL,
                    resolvedTable,
                    CatalogTableModelTsFileExport.TAG_COLUMN_COUNT,
                    CatalogTableModelTsFileExport.INT_ENCODING,
                    CatalogTableModelTsFileExport.FLOAT_ENCODING,
                    CatalogTableModelTsFileExport.STRING_ENCODING,
                    CatalogTableModelTsFileExport.TIME_ENCODING,
                    CatalogTableModelTsFileExport.COMPRESSION,
                    size);
            summary.flush();
        }
        System.out.println(
                "Typed TsFile export: "
                        + tsFile.getAbsolutePath()
                        + " model="
                        + CatalogTableModelTsFileExport.TSFILE_MODEL
                        + " table="
                        + resolvedTable
                        + " compression="
                        + CatalogTableModelTsFileExport.COMPRESSION
                        + " file_size_bytes="
                        + size);
    }

    /**
     * Export a typed CSV to many TsFiles by enumerating per-type encodings (same encoding for all columns of that
     * type within one TsFile). Writes TsFiles under {@code tsFileExportDir} (or {@code {csvDir}/{base}_typed_tsfile_export}
     * when null) and a summary CSV at {@code encodingSummaryPath} (or beside the input CSV when null).
     */
    public static void exportTypedCsvToTsFilesWithEncodingGrid(
            String csvPath, String columnTypesManifestPath) throws Exception {
        exportTypedCsvToTsFilesWithEncodingGrid(csvPath, columnTypesManifestPath, null, null);
    }

    public static void exportTypedCsvToTsFilesWithEncodingGrid(
            String csvPath,
            String columnTypesManifestPath,
            String tsFileExportDirOverride,
            String encodingSummaryPathOverride)
            throws Exception {
        File csvFile = new File(csvPath);
        if (!csvFile.isFile()) {
            throw new FileNotFoundException("CSV not found: " + csvPath);
        }
        File manifestFile = new File(columnTypesManifestPath);
        if (!manifestFile.isFile()) {
            throw new FileNotFoundException("Manifest not found: " + columnTypesManifestPath);
        }

        java.nio.file.Path csvDir = csvFile.toPath().getParent();
        String base = csvFile.getName().substring(0, csvFile.getName().lastIndexOf('.'));
        java.nio.file.Path outDir =
                tsFileExportDirOverride != null && !tsFileExportDirOverride.isBlank()
                        ? java.nio.file.Paths.get(tsFileExportDirOverride)
                        : csvDir.resolve(base + "_typed_tsfile_export");
        Files.createDirectories(outDir);

        TypedExportManifest typedManifest = readTypedExportManifest(columnTypesManifestPath);
        List<ColumnLogicalType> manifest = typedManifest.columns;

        String[] headers;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile, StandardCharsets.UTF_8))) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("CSV is empty");
            }
            headers = headerLine.split(",", -1);
        }

        List<ColumnLogicalType> intCols = new ArrayList<>();
        List<ColumnLogicalType> floatCols = new ArrayList<>();
        List<ColumnLogicalType> strCols = new ArrayList<>();
        for (ColumnLogicalType c : manifest) {
            if (c.index < 0 || c.index >= headers.length) {
                throw new IOException("Manifest column_index out of range: " + c.index);
            }
            switch (c.logicalType) {
                case "integer":
                    intCols.add(c);
                    break;
                case "float":
                    floatCols.add(c);
                    break;
                case "string":
                    strCols.add(c);
                    break;
                default:
                    throw new IOException("Unknown logical type: " + c.logicalType);
            }
        }

        List<TSEncoding> intEncs = intCols.isEmpty() ? Arrays.asList((TSEncoding) null) : supportedEncodingsFor(TSDataType.INT64);
        List<TSEncoding> floatEncs =
                floatCols.isEmpty() ? Arrays.asList((TSEncoding) null) : supportedEncodingsFor(TSDataType.DOUBLE);
        List<TSEncoding> strEncs =
                strCols.isEmpty() ? Arrays.asList((TSEncoding) null) : supportedEncodingsFor(TSDataType.STRING);

        java.nio.file.Path summaryPath =
                encodingSummaryPathOverride != null && !encodingSummaryPathOverride.isBlank()
                        ? java.nio.file.Paths.get(encodingSummaryPathOverride)
                        : csvDir.resolve(base + "_typed_tsfile_encoding_summary.csv");
        if (summaryPath.getParent() != null) {
            Files.createDirectories(summaryPath.getParent());
        }
        long bestSizeBytes = Long.MAX_VALUE;
        String bestTsFileName = null;
        String bestIntEncName = null;
        String bestFloatEncName = null;
        String bestStrEncName = null;
        try (CSVPrinter summary =
                new CSVPrinter(
                        Files.newBufferedWriter(summaryPath, StandardCharsets.UTF_8),
                        CSVFormat.DEFAULT
                                .withHeader(
                                        "int_encoding",
                                        "float_encoding",
                                        "string_encoding",
                                        "file_size_bytes")
                                .withRecordSeparator("\n"))) {

            for (TSEncoding intEnc : intEncs) {
                for (TSEncoding floatEnc : floatEncs) {
                    for (TSEncoding strEnc : strEncs) {
                        String intName = intEnc == null ? "NONE" : intEnc.name();
                        String floatName = floatEnc == null ? "NONE" : floatEnc.name();
                        String strName = strEnc == null ? "NONE" : strEnc.name();
                        String tsName =
                                base
                                        + "_int_"
                                        + intName
                                        + "_float_"
                                        + floatName
                                        + "_str_"
                                        + strName
                                        + ".tsfile";
                        java.nio.file.Path tsPath = outDir.resolve(tsName);

                        try {
                            writeOneTypedTsFile(
                                    csvPath,
                                    headers,
                                    intCols,
                                    floatCols,
                                    strCols,
                                    intEnc,
                                    floatEnc,
                                    strEnc,
                                    tsPath.toFile());
                            long size = Files.exists(tsPath) ? Files.size(tsPath) : 0L;
                            summary.printRecord(intName, floatName, strName, size);
                            if (size < bestSizeBytes
                                    || (size == bestSizeBytes
                                            && bestTsFileName != null
                                            && tsName.compareTo(bestTsFileName) < 0)
                                    || bestTsFileName == null) {
                                bestSizeBytes = size;
                                bestTsFileName = tsName;
                                bestIntEncName = intName;
                                bestFloatEncName = floatName;
                                bestStrEncName = strName;
                            }
                        } catch (Exception e) {
                            // Skip failed encoding combos in the summary CSV.
                        }
                    }
                }
            }
            summary.flush();
        }
        if (bestTsFileName != null) {
            System.out.println(
                    "Smallest typed TsFile by size: "
                            + bestTsFileName
                            + " (int="
                            + bestIntEncName
                            + ", float="
                            + bestFloatEncName
                            + ", string="
                            + bestStrEncName
                            + ", file_size_bytes="
                            + bestSizeBytes
                            + ")");
        } else {
            System.out.println("No successful typed TsFile exports; cannot pick smallest file.");
        }
    }

    /**
     * One encoding grid cell: read {@code csvPath} once, stream each row into {@code tsFile} (no full-table buffer).
     */
    private static void writeOneTypedTsFile(
            String csvPath,
            String[] headers,
            List<ColumnLogicalType> intCols,
            List<ColumnLogicalType> floatCols,
            List<ColumnLogicalType> strCols,
            TSEncoding intEnc,
            TSEncoding floatEnc,
            TSEncoding strEnc,
            File tsFile)
            throws Exception {

        String deviceId = "device_1";
        Path devicePath = new Path(deviceId);

        List<IMeasurementSchema> measurementSchemas = new ArrayList<>();

        for (ColumnLogicalType c : intCols) {
            Map<String, String> props = new HashMap<>();
            props.put(Encoder.MAX_POINT_NUMBER, "0");
            measurementSchemas.add(
                    new MeasurementSchema(
                            measurementIdForColumn(c.index),
                            TSDataType.INT64,
                            intEnc,
                            CompressionType.UNCOMPRESSED,
                            props));
        }
        for (ColumnLogicalType c : floatCols) {
            Map<String, String> props = new HashMap<>();
            props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(c.maxPointNumber));
            measurementSchemas.add(
                    new MeasurementSchema(
                            measurementIdForColumn(c.index),
                            TSDataType.DOUBLE,
                            floatEnc,
                            CompressionType.UNCOMPRESSED,
                            props));
        }
        for (ColumnLogicalType c : strCols) {
            Map<String, String> props = new HashMap<>();
            props.put(Encoder.MAX_POINT_NUMBER, "0");
            measurementSchemas.add(
                    new MeasurementSchema(
                            measurementIdForColumn(c.index),
                            TSDataType.STRING,
                            strEnc,
                            CompressionType.UNCOMPRESSED,
                            props));
        }

        if (measurementSchemas.isEmpty()) {
            throw new IOException("No columns to write (empty manifest selection)");
        }

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath, StandardCharsets.UTF_8));
                TsFileWriter writer = new TsFileWriter(tsFile)) {
            if (br.readLine() == null) {
                throw new IOException("CSV is empty");
            }
            writer.registerTimeseries(devicePath, measurementSchemas);

            String line;
            long rowIndex = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                TSRecord record = new TSRecord(deviceId, rowIndex++);

                for (ColumnLogicalType c : intCols) {
                    if (c.index < values.length) {
                        String raw = normalizeCsvNumericCell(values[c.index]);
                        if (raw.isEmpty()) {
                            continue;
                        }
                        try {
                            long v = Long.parseLong(raw);
                            record.addTuple(new LongDataPoint(measurementIdForColumn(c.index), v));
                        } catch (NumberFormatException ignored) {
                            // skip invalid cell
                        }
                    }
                }
                for (ColumnLogicalType c : floatCols) {
                    if (c.index < values.length) {
                        String raw = normalizeCsvNumericCell(values[c.index]);
                        if (raw.isEmpty()) {
                            continue;
                        }
                        try {
                            double v = Double.parseDouble(raw);
                            if (!ExperimentDataPolicy.keepNumericValue(v)) continue;
                            record.addTuple(new DoubleDataPoint(measurementIdForColumn(c.index), v));
                        } catch (NumberFormatException ignored) {
                            // skip invalid cell
                        }
                    }
                }
                for (ColumnLogicalType c : strCols) {
                    if (c.index < values.length) {
                        String raw = values[c.index];
                        if (raw == null) {
                            continue;
                        }
                        Binary bin = new Binary(raw.getBytes(StandardCharsets.UTF_8));
                        record.addTuple(new StringDataPoint(measurementIdForColumn(c.index), bin));
                    }
                }

                writer.writeRecord(record);
            }
        }
    }

    static void printColumnarEncodingRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> columnIntData,
            ArrayList<ArrayList<Float>> columnFloatData,
            CSVPrinter csvWriter,
            Set<String> existingAlgorithms) throws IOException {
        if (tsType == TSDataType.INT32 && columnIntData != null && !columnIntData.isEmpty()) {
            List<List<Integer>> cols = new ArrayList<>(columnIntData);
            long[] t = new long[4];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                ColumnarEncodings.benchDeltaZigZagVarint(cols, t);
            }
            if (t[0] > 0 && ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                csvWriter.printRecord(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t[0], t[1], t[2], t[3]);
                existingAlgorithms.add(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM);
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                ColumnarEncodings.benchDictionaryInt(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                csvWriter.printRecord("col_dictionary", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_dictionary");
            }
            List<List<Float>> floatView = new ArrayList<>(BenchmarkColumns.integersAsFloatColumns(columnIntData));
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                ColumnarEncodings.benchBssLz4Float(floatView, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                csvWriter.printRecord("col_bss_lz4", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_bss_lz4");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "ts_2diff_bos")) {
                try {
                    Ts2DiffBosImprove.benchIntColumns(cols, t);
                    if (t[0] > 0) {
                        csvWriter.printRecord("ts_2diff_bos", t[0], t[1], t[2], t[3]);
                        existingAlgorithms.add("ts_2diff_bos");
                    }
                } catch (Throwable ignored) {
                    // Wide int deltas can make BOS allocate huge histograms (NegativeArraySizeException).
                }
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "subcolumn")) {
                List<List<Long>> longCols = new ArrayList<>(cols.size());
                for (List<Integer> c : cols) {
                    ArrayList<Long> lc = new ArrayList<>(c.size());
                    for (Integer v : c) {
                        lc.add(v == null ? 0L : v.longValue());
                    }
                    longCols.add(lc);
                }
                SubcolumnCodec.benchSubcolumnLong(longCols, t, null);
                if (t[0] > 0) {
                    csvWriter.printRecord("subcolumn", t[0], t[1], t[2], t[3]);
                    existingAlgorithms.add("subcolumn");
                }
            }
        } else if (tsType == TSDataType.FLOAT && columnFloatData != null && !columnFloatData.isEmpty()) {
            List<List<Float>> cols = new ArrayList<>(columnFloatData);
            long[] t = new long[4];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                ColumnarEncodings.benchDbpFloat32(cols, null, t, null);
            }
            if (t[0] > 0 && ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                csvWriter.printRecord(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t[0], t[1], t[2], t[3]);
                existingAlgorithms.add(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM);
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                ColumnarEncodings.benchDictionaryFloat(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                csvWriter.printRecord("col_dictionary", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_dictionary");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                ColumnarEncodings.benchBssLz4Float(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                csvWriter.printRecord("col_bss_lz4", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_bss_lz4");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "ts_2diff_bos")) {
                try {
                    Ts2DiffBosImprove.benchFloatColumns(cols, t);
                    if (t[0] > 0) {
                        csvWriter.printRecord("ts_2diff_bos", t[0], t[1], t[2], t[3]);
                        existingAlgorithms.add("ts_2diff_bos");
                    }
                } catch (Throwable ignored) {
                    // Same as int path: pathological scaled deltas can overflow BOS histogram allocation.
                }
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "subcolumn")) {
                List<List<Double>> doubleCols = new ArrayList<>(cols.size());
                for (List<Float> c : cols) {
                    ArrayList<Double> dc = new ArrayList<>(c.size());
                    for (Float v : c) {
                        dc.add(v == null ? Double.NaN : v.doubleValue());
                    }
                    doubleCols.add(dc);
                }
                SubcolumnCodec.benchSubcolumnDouble(doubleCols, null, t, null);
                if (t[0] > 0) {
                    csvWriter.printRecord("subcolumn", t[0], t[1], t[2], t[3]);
                    existingAlgorithms.add("subcolumn");
                }
            }
        }
    }

    static void printColumnarEncodingRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> columnIntData,
            ArrayList<ArrayList<Double>> columnFloatData,
            ArrayList<Integer> columnFloatMaxPointData,
            CSVPrinter csvWriter,
            Set<String> existingAlgorithms) throws IOException {
        if (tsType == TSDataType.INT64 && columnIntData != null && !columnIntData.isEmpty()) {
            List<List<Long>> cols = new ArrayList<>(columnIntData);
            long[] t = new long[4];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                ColumnarEncodings.benchDeltaZigZagVarint64(cols, t);
            }
            if (t[0] > 0 && ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                csvWriter.printRecord(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t[0], t[1], t[2], t[3]);
                existingAlgorithms.add(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM);
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                ColumnarEncodings.benchDictionaryInt64(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                csvWriter.printRecord("col_dictionary", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_dictionary");
            }
            List<List<Double>> doubleView =
                    new ArrayList<>(BenchmarkColumns.integersAsDoubleColumns(columnIntData));
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                ColumnarEncodings.benchBssLz4Double64(doubleView, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                csvWriter.printRecord("col_bss_lz4", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_bss_lz4");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "ts_2diff_bos")) {
                try {
                    Ts2DiffBosImprove.benchLongColumns(cols, t);
                    if (t[0] > 0) {
                        csvWriter.printRecord("ts_2diff_bos", t[0], t[1], t[2], t[3]);
                        existingAlgorithms.add("ts_2diff_bos");
                    }
                } catch (Throwable ignored) {
                    // Wide int deltas can make BOS allocate huge histograms (NegativeArraySizeException).
                }
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "subcolumn")) {
                SubcolumnCodec.benchSubcolumnLong(cols, t, null);
                if (t[0] > 0) {
                    csvWriter.printRecord("subcolumn", t[0], t[1], t[2], t[3]);
                    existingAlgorithms.add("subcolumn");
                }
            }
        } else if (tsType == TSDataType.DOUBLE && columnFloatData != null && !columnFloatData.isEmpty()) {
            List<List<Double>> cols = new ArrayList<>(columnFloatData);
            long[] t = new long[4];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                ColumnarEncodings.benchDbpFloat64(cols, columnFloatMaxPointData, t, null);
            }
            if (t[0] > 0 && ColumnarEncodings.shouldRunDeltaBinaryPacked(existingAlgorithms)) {
                csvWriter.printRecord(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t[0], t[1], t[2], t[3]);
                existingAlgorithms.add(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM);
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                ColumnarEncodings.benchDictionaryDouble64(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_dictionary")) {
                csvWriter.printRecord("col_dictionary", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_dictionary");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                ColumnarEncodings.benchBssLz4Double64(cols, t);
            }
            if (t[0] > 0 && !AlgorithmAllowList.shouldSkip(existingAlgorithms, "col_bss_lz4")) {
                csvWriter.printRecord("col_bss_lz4", t[0], t[1], t[2], t[3]);
                existingAlgorithms.add("col_bss_lz4");
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "ts_2diff_bos")) {
                try {
                    Ts2DiffBosImprove.benchDoubleColumns(cols, columnFloatMaxPointData, t);
                    if (t[0] > 0) {
                        csvWriter.printRecord("ts_2diff_bos", t[0], t[1], t[2], t[3]);
                        existingAlgorithms.add("ts_2diff_bos");
                    }
                } catch (Throwable ignored) {
                    // Same as int path: pathological scaled deltas can overflow BOS histogram allocation.
                }
            }
            if (!AlgorithmAllowList.shouldSkip(existingAlgorithms, "subcolumn")) {
                SubcolumnCodec.benchSubcolumnDouble(cols, columnFloatMaxPointData, t, null);
                if (t[0] > 0) {
                    csvWriter.printRecord("subcolumn", t[0], t[1], t[2], t[3]);
                    existingAlgorithms.add("subcolumn");
                }
            }
        }
    }

    static void printColumnarEncodingRows64Accumulate(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> columnIntData,
            ArrayList<ArrayList<Double>> columnFloatData,
            ArrayList<Integer> columnFloatMaxPointData,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        if (tsType == TSDataType.INT64 && columnIntData != null && !columnIntData.isEmpty()) {
            List<List<Long>> cols = new ArrayList<>(columnIntData);
            long[] t = new long[4];
            double[] loss = new double[2];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(skipAlgorithms)) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchDeltaZigZagVarint64(cols, t, false, loss);
                totals.merge(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t, loss[0], loss[1]);
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "col_dictionary")) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchDictionaryInt64(cols, t, loss);
                totals.merge("col_dictionary", t, loss[0], loss[1]);
            }
            List<List<Double>> doubleView =
                    new ArrayList<>(BenchmarkColumns.integersAsDoubleColumns(columnIntData));
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "col_bss_lz4")) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchBssLz4Double64(doubleView, t, loss);
                totals.merge("col_bss_lz4", t, loss[0], loss[1]);
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "ts_2diff_bos")) {
                try {
                    java.util.Arrays.fill(loss, 0);
                    Ts2DiffBosImprove.benchLongColumns(cols, t, loss);
                    totals.merge("ts_2diff_bos", t, loss[0], loss[1]);
                } catch (Throwable ignored) {
                    // skip patch
                }
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "subcolumn")) {
                java.util.Arrays.fill(loss, 0);
                SubcolumnCodec.benchSubcolumnLong(cols, t, loss);
                totals.merge("subcolumn", t, loss[0], loss[1]);
            }
        } else if (tsType == TSDataType.DOUBLE && columnFloatData != null && !columnFloatData.isEmpty()) {
            List<List<Double>> cols = new ArrayList<>(columnFloatData);
            long[] t = new long[4];
            double[] loss = new double[2];
            if (ColumnarEncodings.shouldRunDeltaBinaryPacked(skipAlgorithms)) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchDbpFloat64(cols, columnFloatMaxPointData, t, loss);
                totals.merge(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM, t, loss[0], loss[1]);
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "col_dictionary")) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchDictionaryDouble64(cols, t, loss);
                totals.merge("col_dictionary", t, loss[0], loss[1]);
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "col_bss_lz4")) {
                java.util.Arrays.fill(loss, 0);
                ColumnarEncodings.benchBssLz4Double64(cols, t, loss);
                totals.merge("col_bss_lz4", t, loss[0], loss[1]);
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "ts_2diff_bos")) {
                try {
                    java.util.Arrays.fill(loss, 0);
                    Ts2DiffBosImprove.benchDoubleColumns(cols, columnFloatMaxPointData, t, loss);
                    totals.merge("ts_2diff_bos", t, loss[0], loss[1]);
                } catch (Throwable ignored) {
                    // skip patch
                }
            }
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "subcolumn")) {
                java.util.Arrays.fill(loss, 0);
                SubcolumnCodec.benchSubcolumnDouble(cols, columnFloatMaxPointData, t, loss);
                totals.merge("subcolumn", t, loss[0], loss[1]);
            }
        }
    }
}
