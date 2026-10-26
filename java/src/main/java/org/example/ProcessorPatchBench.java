package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Catalog CSV benchmark: read up to {@link #patchRowLimit()} rows per patch, accumulate metrics.
 * Per-column max decimal places come from {@code {dataset}_column_info.csv} (no re-scan of full file).
 */
final class ProcessorPatchBench {
    private static final int DEFAULT_PATCH_ROWS = 10_000;
    private static final class ColumnInfoMeta {
        final String logicalType;
        final int maxPoint;

        ColumnInfoMeta(String logicalType, int maxPoint) {
            this.logicalType = logicalType == null ? "" : logicalType.trim().toLowerCase();
            this.maxPoint = Math.max(0, maxPoint);
        }
    }

    private ProcessorPatchBench() {}

    static int patchRowLimit() {
        String raw = System.getenv("WEB_COMPRESSION_CSV_PATCH_ROWS");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_PATCH_ROWS;
        }
        try {
            int n = Integer.parseInt(raw.trim());
            return n > 0 ? n : DEFAULT_PATCH_ROWS;
        } catch (NumberFormatException e) {
            return DEFAULT_PATCH_ROWS;
        }
    }

    private static boolean patchBenchVerbose() {
        String raw = System.getenv("WEB_COMPRESSION_PATCH_BENCH_VERBOSE");
        if (raw == null || raw.isBlank()) {
            return false;
        }
        String v = raw.trim().toLowerCase();
        return v.equals("1") || v.equals("true") || v.equals("yes") || v.equals("on");
    }

    static void run(
            String csvPath,
            String dataType,
            String outputReportPath,
            String columnInfoPath,
            Integer columnIndex) {
        try {
            if ("1".equals(System.getenv("WEB_COMPRESSION_STATICDATA_REGEN"))
                    && new File(outputReportPath).isFile()) {
                if (!new File(outputReportPath).delete()) {
                    System.err.println("Warning: could not delete " + outputReportPath + " for regen");
                }
            }
            if (shouldUseMultiColumnReger(dataType, columnIndex)) {
                runMultiColumnReger(csvPath, dataType, outputReportPath, columnInfoPath);
                return;
            }
            if (shouldUseMultiColumnNativeBytes(dataType, columnIndex)) {
                runMultiColumnNativeBytes(csvPath, dataType, outputReportPath, columnInfoPath);
                return;
            }
            if (shouldUseMultiColumnGrouping(dataType, columnIndex)) {
                runMultiColumnGrouping(csvPath, dataType, outputReportPath, columnInfoPath);
                return;
            }
            boolean appendMode = new File(outputReportPath).isFile();
            Set<String> existingAlgorithms = new LinkedHashSet<>(AlgorithmAllowList.fromIncrementalSkipEnv());
            if (!AlgorithmAllowList.hasAllowlist()) {
                existingAlgorithms.addAll(AlgorithmAllowList.fromCsvReport(Path.of(outputReportPath)));
            }

            try (BufferedReader br = new BufferedReader(new FileReader(csvPath));
                    CSVPrinter csvWriter =
                            new CSVPrinter(
                                    new FileWriter(outputReportPath, StandardCharsets.UTF_8, appendMode),
                                    CSVFormat.DEFAULT.withRecordSeparator("\n"))) {
                if (!appendMode) {
                    csvWriter.printRecord(
                            "column_index",
                            "algorithm",
                            "originalSize",
                            "compressedSize",
                            "compressionTime",
                            "decompressionTime",
                            "isLossy",
                            "lossRate");
                }
                String headerLine = br.readLine();
                if (headerLine == null) {
                    System.err.println("CSV file is empty");
                    return;
                }
                String[] headers = headerLine.split(",", -1);
                if (headers.length > 1) {
                    System.err.println(
                            "Warning: ProcessorPatchBench now treats input as single-column CSV; extra columns are ignored.");
                }
                int outColumnIndex = columnIndex == null ? 0 : columnIndex.intValue();
                Map<Integer, ColumnInfoMeta> columnInfoByIndex = loadColumnInfoByIndex(columnInfoPath);
                boolean isString = "string".equalsIgnoreCase(dataType);
                TSDataType tsType = null;
                if (!isString) {
                    tsType =
                            "int".equalsIgnoreCase(dataType)
                                    ? TSDataType.INT64
                                    : TSDataType.DOUBLE;
                }
                ColumnInfoMeta meta = columnInfoByIndex.get(outColumnIndex);
                int floatMaxPoint = meta == null ? 0 : meta.maxPoint;
                if (meta != null && !logicalTypeMatchesDataType(meta.logicalType, dataType)) {
                    System.err.println(
                            "Warning: column_info logical_type="
                                    + meta.logicalType
                                    + " does not match --data-type="
                                    + dataType
                                    + " for column_index="
                                    + outColumnIndex);
                }

                BenchmarkTotals totals = new BenchmarkTotals();
                int patchRows = patchRowLimit();
                int patchIndex = 0;
                long rowsProcessed = 0;
                boolean done = false;
                while (!done) {
                    ArrayList<String> patchColumn = new ArrayList<>();
                    int rowsInPatch = 0;
                    String line;
                    while (rowsInPatch < patchRows && (line = br.readLine()) != null) {
                        String[] values = line.split(",", -1);
                        patchColumn.add(values.length > 0 ? values[0] : "");
                        rowsInPatch++;
                    }
                    if (rowsInPatch == 0) {
                        break;
                    }
                    if (rowsInPatch < patchRows) {
                        done = true;
                    }
                    patchIndex++;
                    rowsProcessed += rowsInPatch;
                    System.out.printf(
                            "ProcessorPatchBench patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=%d)%n",
                            patchIndex,
                            rowsInPatch,
                            patchRows,
                            rowsProcessed,
                            dataType,
                            outColumnIndex);
                    if (isString) {
                        accumulateStringPatch(patchColumn, totals, existingAlgorithms);
                    } else {
                        accumulatePatch(
                                patchColumn,
                                tsType,
                                floatMaxPoint,
                                totals,
                                existingAlgorithms);
                    }
                }
                flushTotals(totals, csvWriter, existingAlgorithms, outColumnIndex);
                csvWriter.flush();
            }
            System.out.println("Compression results written to " + outputReportPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Map<Integer, ColumnInfoMeta> loadColumnInfoByIndex(String columnInfoPath)
            throws Exception {
        Map<Integer, ColumnInfoMeta> out = new HashMap<>();
        if (columnInfoPath == null || columnInfoPath.isBlank()) {
            return out;
        }
        Path path = Path.of(columnInfoPath);
        if (!Files.isRegularFile(path)) {
            return out;
        }
        try (Reader fr = Files.newBufferedReader(path, StandardCharsets.UTF_8);
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
                int idx;
                try {
                    idx = Integer.parseInt(rec.get("column_index"));
                } catch (NumberFormatException ignored) {
                    continue;
                }
                String logical = rec.get("logical_type");
                int mp = 0;
                try {
                    mp = Integer.parseInt(rec.get("max_point_number"));
                } catch (NumberFormatException ignored) {
                    // keep 0
                }
                out.put(idx, new ColumnInfoMeta(logical, mp));
            }
        }
        return out;
    }

    private static boolean logicalTypeMatchesDataType(String logicalType, String dataType) {
        String l = logicalType == null ? "" : logicalType.trim().toLowerCase();
        if ("int".equalsIgnoreCase(dataType)) {
            return l.equals("integer") || l.equals("int") || l.equals("long");
        }
        if ("float".equalsIgnoreCase(dataType)) {
            return l.equals("float") || l.equals("double");
        }
        if ("string".equalsIgnoreCase(dataType)) {
            return l.equals("string");
        }
        return true;
    }

    private static boolean shouldUseMultiColumnReger(String dataType, Integer columnIndex) {
        if (columnIndex != null || "string".equalsIgnoreCase(dataType)) {
            return false;
        }
        Set<String> allow = AlgorithmAllowList.fromAllowlistEnv();
        return allow.size() == 1 && allow.contains("reger");
    }

    private static boolean shouldUseMultiColumnNativeBytes(String dataType, Integer columnIndex) {
        if (columnIndex != null) {
            return false;
        }
        if (!"int".equalsIgnoreCase(dataType)
                && !"float".equalsIgnoreCase(dataType)
                && !"string".equalsIgnoreCase(dataType)) {
            return false;
        }
        Set<String> allow = AlgorithmAllowList.fromAllowlistEnv();
        return !allow.isEmpty() && NativeLzFamilyBench.algorithms().containsAll(allow);
    }

    private static boolean shouldUseMultiColumnGrouping(String dataType, Integer columnIndex) {
        if (columnIndex != null) {
            return false;
        }
        if (!"int".equalsIgnoreCase(dataType) && !"float".equalsIgnoreCase(dataType)) {
            return false;
        }
        Set<String> allow = AlgorithmAllowList.fromAllowlistEnv();
        return allow.size() == 1 && allow.contains(GroupingCodec.ROW_NAME);
    }

    private static void runMultiColumnGrouping(
            String csvPath, String dataType, String outputReportPath, String columnInfoPath)
            throws Exception {
        boolean appendMode = new File(outputReportPath).isFile();
        Set<String> existingAlgorithms = new LinkedHashSet<>(AlgorithmAllowList.fromIncrementalSkipEnv());
        if (!AlgorithmAllowList.hasAllowlist()) {
            existingAlgorithms.addAll(AlgorithmAllowList.fromCsvReport(Path.of(outputReportPath)));
        }
        Map<Integer, ColumnInfoMeta> columnInfoByIndex = loadColumnInfoByIndex(columnInfoPath);
        BenchmarkTotals totals = new BenchmarkTotals();
        int patchRows = patchRowLimit();
        boolean verbose = patchBenchVerbose();
        int patchIndex = 0;
        long rowsProcessed = 0;

        try (Reader fr = Files.newBufferedReader(Path.of(csvPath), StandardCharsets.UTF_8);
                CSVParser parser =
                        CSVFormat.DEFAULT.builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(false)
                                .build()
                                .parse(fr);
                CSVPrinter csvWriter =
                        new CSVPrinter(
                                new FileWriter(outputReportPath, StandardCharsets.UTF_8, appendMode),
                                CSVFormat.DEFAULT.withRecordSeparator("\n"))) {
            if (!appendMode) {
                csvWriter.printRecord(
                        "column_index",
                        "algorithm",
                        "originalSize",
                        "compressedSize",
                        "compressionTime",
                        "decompressionTime",
                        "isLossy",
                        "lossRate");
            }

            List<String> headers = parser.getHeaderNames();
            List<Integer> valueIndexes = new ArrayList<>();
            for (int i = 0; i < headers.size(); i++) {
                String header = headers.get(i) == null ? "" : headers.get(i).trim();
                if ("__reger_time__".equals(header)) {
                    continue;
                }
                ColumnInfoMeta meta = columnInfoByIndex.get(i);
                if (meta != null && !logicalTypeMatchesDataType(meta.logicalType, dataType)) {
                    continue;
                }
                valueIndexes.add(i);
            }
            if (valueIndexes.isEmpty()) {
                System.err.println("No value columns for Java Grouping in " + csvPath);
                return;
            }

            List<List<String>> patchColumns = new ArrayList<>();
            for (int ignored : valueIndexes) {
                patchColumns.add(new ArrayList<>());
            }
            int rowsInPatch = 0;
            for (CSVRecord rec : parser) {
                for (int i = 0; i < valueIndexes.size(); i++) {
                    patchColumns.get(i).add(cellAt(rec, valueIndexes.get(i)));
                }
                rowsInPatch++;
                if (rowsInPatch >= patchRows) {
                    long rowOffset = rowsProcessed;
                    patchIndex++;
                    rowsProcessed += rowsInPatch;
                    if (verbose) {
                        System.out.printf(
                                "ProcessorPatchBench grouping patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                                patchIndex,
                                rowsInPatch,
                                patchRows,
                                rowsProcessed,
                                dataType);
                    }
                    accumulateGroupingPatch(dataType, patchColumns, rowOffset, totals);
                    clearPatch(patchColumns, new ArrayList<>());
                    rowsInPatch = 0;
                }
            }
            if (rowsInPatch > 0) {
                long rowOffset = rowsProcessed;
                patchIndex++;
                rowsProcessed += rowsInPatch;
                if (verbose) {
                    System.out.printf(
                            "ProcessorPatchBench grouping patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                            patchIndex,
                            rowsInPatch,
                            patchRows,
                            rowsProcessed,
                            dataType);
                }
                accumulateGroupingPatch(dataType, patchColumns, rowOffset, totals);
            }
            flushTotals(totals, csvWriter, existingAlgorithms, -1);
            csvWriter.flush();
        }
        System.out.println("Compression results written to " + outputReportPath);
    }

    private static void accumulateGroupingPatch(
            String dataType, List<List<String>> patchColumns, long rowOffset, BenchmarkTotals totals) {
        List<GroupingCodec.ColumnStream> streams = new ArrayList<>();
        for (List<String> col : patchColumns) {
            GroupingCodec.ColumnStream stream =
                    "int".equalsIgnoreCase(dataType)
                            ? GroupingCodec.intColumnStream(col, rowOffset)
                            : GroupingCodec.doubleColumnStream(col, rowOffset);
            if (!stream.values.isEmpty()) {
                streams.add(stream);
            }
        }
        if (streams.isEmpty()) {
            return;
        }
        long[] result = new long[4];
        GroupingCodec.benchStreams(streams, result);
        totals.merge(GroupingCodec.ROW_NAME, result);
    }

    private static void runMultiColumnNativeBytes(
            String csvPath, String dataType, String outputReportPath, String columnInfoPath)
            throws Exception {
        boolean appendMode = new File(outputReportPath).isFile();
        Set<String> existingAlgorithms = new LinkedHashSet<>(AlgorithmAllowList.fromIncrementalSkipEnv());
        if (!AlgorithmAllowList.hasAllowlist()) {
            existingAlgorithms.addAll(AlgorithmAllowList.fromCsvReport(Path.of(outputReportPath)));
        }
        BenchmarkTotals totals = new BenchmarkTotals();
        int patchRows = patchRowLimit();
        boolean verbose = patchBenchVerbose();
        int patchIndex = 0;
        long rowsProcessed = 0;

        try (Reader fr = Files.newBufferedReader(Path.of(csvPath), StandardCharsets.UTF_8);
                CSVParser parser =
                        CSVFormat.DEFAULT.builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(false)
                                .build()
                                .parse(fr);
                CSVPrinter csvWriter =
                        new CSVPrinter(
                                new FileWriter(outputReportPath, StandardCharsets.UTF_8, appendMode),
                                CSVFormat.DEFAULT.withRecordSeparator("\n"))) {
            if (!appendMode) {
                csvWriter.printRecord(
                        "column_index",
                        "algorithm",
                        "originalSize",
                        "compressedSize",
                        "compressionTime",
                        "decompressionTime",
                        "isLossy",
                        "lossRate");
            }

            List<String> headers = parser.getHeaderNames();
            if (headers.isEmpty()) {
                System.err.println("CSV file has no headers: " + csvPath);
                return;
            }
            List<List<String>> patchColumns = new ArrayList<>();
            for (int i = 0; i < headers.size(); i++) {
                patchColumns.add(new ArrayList<>());
            }
            int rowsInPatch = 0;
            for (CSVRecord rec : parser) {
                for (int i = 0; i < headers.size(); i++) {
                    patchColumns.get(i).add(cellAt(rec, i));
                }
                rowsInPatch++;
                if (rowsInPatch >= patchRows) {
                    patchIndex++;
                    rowsProcessed += rowsInPatch;
                    if (verbose) {
                        System.out.printf(
                                "ProcessorPatchBench patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                                patchIndex,
                                rowsInPatch,
                                patchRows,
                                rowsProcessed,
                                dataType);
                    }
                    accumulateNativeBytePatch(dataType, patchColumns, totals);
                    clearPatch(patchColumns, new ArrayList<>());
                    rowsInPatch = 0;
                }
            }
            if (rowsInPatch > 0) {
                patchIndex++;
                rowsProcessed += rowsInPatch;
                if (verbose) {
                    System.out.printf(
                            "ProcessorPatchBench patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                            patchIndex,
                            rowsInPatch,
                            patchRows,
                            rowsProcessed,
                            dataType);
                }
                accumulateNativeBytePatch(dataType, patchColumns, totals);
            }
            flushTotals(totals, csvWriter, existingAlgorithms, -1);
            csvWriter.flush();
        }
        System.out.println("Compression results written to " + outputReportPath);
    }

    private static void accumulateNativeBytePatch(
            String dataType, List<List<String>> patchColumns, BenchmarkTotals totals)
            throws Exception {
        ArrayList<byte[]> columnByteData = new ArrayList<>();
        for (List<String> rawCol : patchColumns) {
            java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
            java.nio.ByteBuffer le =
                    java.nio.ByteBuffer.allocate(BenchmarkWireFormat.INT_BYTES)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
            if ("int".equalsIgnoreCase(dataType)) {
                for (String cell : rawCol) {
                    String t = normalizeCell(cell);
                    if (t.isEmpty()) {
                        continue;
                    }
                    try {
                        Processor.appendColumnLongLe(baos, Long.parseLong(t), le);
                    } catch (NumberFormatException ignored) {
                        // skip invalid numeric cells
                    }
                }
            } else if ("float".equalsIgnoreCase(dataType)) {
                for (String cell : rawCol) {
                    String t = normalizeCell(cell);
                    if (t.isEmpty()) {
                        continue;
                    }
                    try {
                        double value = Double.parseDouble(t);
                        if (!ExperimentDataPolicy.keepNumericValue(value)) continue;
                        Processor.appendColumnDoubleLe(baos, value, le);
                    } catch (NumberFormatException ignored) {
                        // skip invalid numeric cells
                    }
                }
            } else {
                for (String cell : rawCol) {
                    baos.write(normalizeCell(cell).getBytes(StandardCharsets.UTF_8));
                }
            }
            if (baos.size() > 0) {
                columnByteData.add(baos.toByteArray());
            }
        }
        if (!columnByteData.isEmpty()) {
            accumulateGenericByteCompression(columnByteData, totals, AlgorithmAllowList.fromIncrementalSkipEnv());
        }
    }

    private static void runMultiColumnReger(
            String csvPath, String dataType, String outputReportPath, String columnInfoPath)
            throws Exception {
        boolean appendMode = new File(outputReportPath).isFile();
        Set<String> existingAlgorithms = new LinkedHashSet<>(AlgorithmAllowList.fromIncrementalSkipEnv());
        if (!AlgorithmAllowList.hasAllowlist()) {
            existingAlgorithms.addAll(AlgorithmAllowList.fromCsvReport(Path.of(outputReportPath)));
        }
        Map<Integer, ColumnInfoMeta> columnInfoByIndex = loadColumnInfoByIndex(columnInfoPath);
        BenchmarkTotals totals = new BenchmarkTotals();
        int patchRows = patchRowLimit();
        boolean verbose = patchBenchVerbose();
        int patchIndex = 0;
        long rowsProcessed = 0;

        try (Reader fr = Files.newBufferedReader(Path.of(csvPath), StandardCharsets.UTF_8);
                CSVParser parser =
                        CSVFormat.DEFAULT.builder()
                                .setHeader()
                                .setSkipHeaderRecord(true)
                                .setIgnoreEmptyLines(false)
                                .build()
                                .parse(fr);
                CSVPrinter csvWriter =
                        new CSVPrinter(
                                new FileWriter(outputReportPath, StandardCharsets.UTF_8, appendMode),
                                CSVFormat.DEFAULT.withRecordSeparator("\n"))) {
            if (!appendMode) {
                csvWriter.printRecord(
                        "column_index",
                        "algorithm",
                        "originalSize",
                        "compressedSize",
                        "compressionTime",
                        "decompressionTime",
                        "isLossy",
                        "lossRate");
            }

            List<String> headers = parser.getHeaderNames();
            int timeIndex = -1;
            List<Integer> valueIndexes = new ArrayList<>();
            List<Integer> valueMaxPoints = new ArrayList<>();
            for (int i = 0; i < headers.size(); i++) {
                String header = headers.get(i) == null ? "" : headers.get(i).trim();
                if ("__reger_time__".equals(header)) {
                    timeIndex = i;
                    continue;
                }
                valueIndexes.add(i);
                ColumnInfoMeta meta = columnInfoByIndex.get(i);
                valueMaxPoints.add(meta == null ? 0 : meta.maxPoint);
            }
            if (valueIndexes.isEmpty()) {
                System.err.println("No value columns for Java REGER in " + csvPath);
                return;
            }

            List<List<String>> patchColumns = new ArrayList<>();
            for (int ignored : valueIndexes) {
                patchColumns.add(new ArrayList<>());
            }
            List<Long> patchTimes = new ArrayList<>();
            int rowsInPatch = 0;
            long rowIndex = 0;
            for (CSVRecord rec : parser) {
                if (timeIndex >= 0) {
                    patchTimes.add(parseRegerTimeCell(cellAt(rec, timeIndex), rowIndex));
                }
                for (int i = 0; i < valueIndexes.size(); i++) {
                    patchColumns.get(i).add(cellAt(rec, valueIndexes.get(i)));
                }
                rowsInPatch++;
                rowIndex++;
                if (rowsInPatch >= patchRows) {
                    patchIndex++;
                rowsProcessed += rowsInPatch;
                if (verbose) {
                    System.out.printf(
                            "ProcessorPatchBench patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                            patchIndex,
                            rowsInPatch,
                            patchRows,
                            rowsProcessed,
                            dataType);
                }
                accumulateRegerPatch(dataType, patchColumns, patchTimes, valueMaxPoints, totals);
                clearPatch(patchColumns, patchTimes);
                rowsInPatch = 0;
                }
            }
            if (rowsInPatch > 0) {
                patchIndex++;
                rowsProcessed += rowsInPatch;
                if (verbose) {
                    System.out.printf(
                            "ProcessorPatchBench patch %d: %d rows (limit=%d, total=%d, data_type=%s, column_index=-1)%n",
                            patchIndex,
                            rowsInPatch,
                            patchRows,
                            rowsProcessed,
                            dataType);
                }
                accumulateRegerPatch(dataType, patchColumns, patchTimes, valueMaxPoints, totals);
            }
            flushTotals(totals, csvWriter, existingAlgorithms, -1);
            csvWriter.flush();
        }
        System.out.println("Compression results written to " + outputReportPath);
    }

    private static void clearPatch(List<List<String>> patchColumns, List<Long> patchTimes) {
        for (List<String> col : patchColumns) {
            col.clear();
        }
        patchTimes.clear();
    }

    private static String cellAt(CSVRecord rec, int index) {
        return index >= 0 && index < rec.size() ? rec.get(index) : "";
    }

    private static long parseRegerTimeCell(String raw, long rowIndex) {
        String t = normalizeCell(raw);
        if (t.isEmpty()) {
            return rowIndex;
        }
        try {
            double numeric = Double.parseDouble(t);
            if (Double.isFinite(numeric)) {
                if (Math.abs(numeric) < 100_000_000_000.0) {
                    return Math.round(numeric * 1_000_000.0);
                }
                return Math.round(numeric);
            }
        } catch (NumberFormatException ignored) {
            // Non-numeric direct inputs use row order.
        }
        return rowIndex;
    }

    private static void accumulateRegerPatch(
            String dataType,
            List<List<String>> patchColumns,
            List<Long> patchTimes,
            List<Integer> valueMaxPoints,
            BenchmarkTotals totals) {
        List<List<Long>> intColumns = new ArrayList<>();
        if ("int".equalsIgnoreCase(dataType)) {
            for (List<String> rawCol : patchColumns) {
                List<Long> col = new ArrayList<>(rawCol.size());
                for (String cell : rawCol) {
                    String t = normalizeCell(cell);
                    if (t.isEmpty()) {
                        continue;
                    }
                    try {
                        col.add(Long.parseLong(t));
                    } catch (NumberFormatException ignored) {
                        // skip invalid numeric cells
                    }
                }
                if (!col.isEmpty()) {
                    intColumns.add(col);
                }
            }
        } else {
            for (int i = 0; i < patchColumns.size(); i++) {
                List<Double> doubles = new ArrayList<>(patchColumns.get(i).size());
                for (String cell : patchColumns.get(i)) {
                    String t = normalizeCell(cell);
                    if (t.isEmpty()) {
                        continue;
                    }
                    try {
                        double value = Double.parseDouble(t);
                        if (!ExperimentDataPolicy.keepNumericValue(value)) continue;
                        doubles.add(value);
                    } catch (NumberFormatException ignored) {
                        // skip invalid numeric cells
                    }
                }
                List<Long> scaled = scaleFloatColumnToInt(doubles, valueMaxPoints.get(i));
                if (scaled != null && !scaled.isEmpty()) {
                    intColumns.add(scaled);
                }
            }
        }
        if (intColumns.isEmpty()) {
            return;
        }
        long[] result = new long[4];
        double[] loss = new double[2];
        RegerCodec.benchInt64ColumnsReger(
                intColumns, patchTimes.isEmpty() ? null : patchTimes, result, loss);
        totals.merge("reger", result, loss[0], loss[1]);
    }

    private static List<Long> scaleFloatColumnToInt(List<Double> values, int maxPoint) {
        if (values == null || values.isEmpty()) {
            return null;
        }
        int startScale = Math.max(0, maxPoint);
        for (int scale = startScale; scale >= 0; scale--) {
            double factor = Math.pow(10.0, scale);
            List<Long> out = new ArrayList<>(values.size());
            boolean ok = true;
            for (Double value : values) {
                double scaled = value.doubleValue() * factor;
                if (!Double.isFinite(scaled)
                        || scaled < Long.MIN_VALUE
                        || scaled > Long.MAX_VALUE) {
                    ok = false;
                    break;
                }
                out.add(Math.round(scaled));
            }
            if (ok) {
                return out;
            }
        }
        return null;
    }

    /** UTF-8 concatenation of normalized cell values (generic byte compressors only). */
    private static void accumulateStringPatch(
            ArrayList<String> patchColumn,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws Exception {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        for (String v : patchColumn) {
            String t = normalizeCell(v);
            baos.write(t.getBytes(StandardCharsets.UTF_8));
        }
        if (baos.size() <= 0) {
            return;
        }
        ArrayList<byte[]> columnByteData = new ArrayList<>();
        columnByteData.add(baos.toByteArray());
        accumulateGenericByteCompression(columnByteData, totals, skipAlgorithms);
    }

    private static void accumulateGenericByteCompression(
            List<byte[]> columnByteData,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws Exception {
        for (String algorithm : NativeLzFamilyBench.algorithms()) {
            if (AlgorithmAllowList.shouldSkip(skipAlgorithms, algorithm)) {
                continue;
            }
            long[] acc = new long[4];
            double[] lossAcc = new double[2];
            NativeLzFamilyBench.benchByteColumns(columnByteData, algorithm, acc, lossAcc);
            totals.merge(algorithm, acc, lossAcc[0], lossAcc[1]);
        }
    }

    private static void accumulatePatch(
            ArrayList<String> patchColumn,
            TSDataType tsType,
            int floatMaxPoint,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws Exception {
        ArrayList<byte[]> columnByteData = new ArrayList<>();
        ArrayList<ArrayList<Long>> columnIntData = new ArrayList<>();
        ArrayList<ArrayList<Double>> columnFloatData = new ArrayList<>();
        ArrayList<Integer> columnFloatMaxPointData = new ArrayList<>();

        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        java.nio.ByteBuffer le =
                java.nio.ByteBuffer.allocate(BenchmarkWireFormat.INT_BYTES)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        ArrayList<Long> intList = new ArrayList<>();
        ArrayList<Double> floatList = new ArrayList<>();

        if (tsType == TSDataType.INT64) {
            for (String v : patchColumn) {
                String t = normalizeCell(v);
                if (!t.isEmpty()) {
                    try {
                        long intValue = Long.parseLong(t);
                        Processor.appendColumnLongLe(baos, intValue, le);
                        intList.add(intValue);
                    } catch (NumberFormatException ignored) {
                        // skip
                    }
                }
            }
        } else {
            for (String v : patchColumn) {
                String t = normalizeCell(v);
                if (!t.isEmpty()) {
                    try {
                        double floatValue = Double.parseDouble(t);
                        if (!ExperimentDataPolicy.keepNumericValue(floatValue)) continue;
                        Processor.appendColumnDoubleLe(baos, floatValue, le);
                        floatList.add(floatValue);
                    } catch (NumberFormatException ignored) {
                        // skip
                    }
                }
            }
        }

        if (baos.size() > 0) {
            columnByteData.add(baos.toByteArray());
            if (tsType == TSDataType.INT64) {
                columnIntData.add(intList);
            } else {
                columnFloatData.add(floatList);
                columnFloatMaxPointData.add(floatMaxPoint);
            }
        } else {
            return;
        }

        accumulateGenericByteCompression(columnByteData, totals, skipAlgorithms);
        Set<String> downstreamSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
        downstreamSkipAlgorithms.add("blosc_blosclz");
        downstreamSkipAlgorithms.add("blosc2_blosclz");
        downstreamSkipAlgorithms.add("fpzip");
        downstreamSkipAlgorithms.add("fpcompress_dpratio");
        downstreamSkipAlgorithms.add("fpcompress_dpspeed");
        downstreamSkipAlgorithms.add("elf_on_chimp");
        downstreamSkipAlgorithms.add("elf_plus");
        downstreamSkipAlgorithms.add("camel");

        List<TSEncoding> encodingList = new ArrayList<>();
        encodingList.add(TSEncoding.TS_2DIFF);
        encodingList.add(TSEncoding.RLE);
        encodingList.add(TSEncoding.GORILLA);
        encodingList.add(TSEncoding.CHIMP);
        encodingList.add(TSEncoding.SPRINTZ);
        encodingList.add(TSEncoding.RLBE);
        for (TSEncoding encoding : encodingList) {
            String rowName = encoding.toString();
            if (AlgorithmAllowList.shouldSkip(skipAlgorithms, rowName)) {
                continue;
            }
            long totalOriginal = 0;
            long totalEncoded = 0;
            long totalCompTime = 0;
            long totalDecompTime = 0;
            double lossErr = 0;
            double lossRef = 0;
            double[] lossAcc = new double[2];
            for (int i = 0; i < columnByteData.size(); i++) {
                long[] result = new long[4];
                try {
                    java.util.Arrays.fill(lossAcc, 0);
                    if (tsType == TSDataType.INT64) {
                        CompressionUtils.testEncoding(
                                tsType, encoding, columnIntData.get(i), result, lossAcc);
                    } else {
                        int maxPoint =
                                i < columnFloatMaxPointData.size()
                                        ? columnFloatMaxPointData.get(i)
                                        : 0;
                        CompressionUtils.testEncoding(
                                tsType, encoding, columnFloatData.get(i), maxPoint, result, lossAcc);
                    }
                    totalOriginal += result[0];
                    totalEncoded += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                    lossErr += lossAcc[0];
                    lossRef += lossAcc[1];
                } catch (Exception ignored) {
                    // skip
                }
            }
            totals.merge(rowName, totalOriginal, totalEncoded, totalCompTime, totalDecompTime, lossErr, lossRef);
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "CHIMP128")) {
            long totalOriginal = 0;
            long totalEncoded = 0;
            long totalCompTime = 0;
            long totalDecompTime = 0;
            double lossErr = 0;
            double lossRef = 0;
            double[] lossAcc = new double[2];
            boolean verbose = patchBenchVerbose();
            for (int i = 0; i < columnByteData.size(); i++) {
                long[] result = new long[4];
                try {
                    java.util.Arrays.fill(lossAcc, 0);
                    if (tsType == TSDataType.INT64) {
                        CompressionUtils.testEncoding(
                                tsType, TSEncoding.CHIMP, columnIntData.get(i), result, lossAcc);
                    } else {
                        int maxPoint =
                                i < columnFloatMaxPointData.size()
                                        ? columnFloatMaxPointData.get(i)
                                        : 0;
                        CompressionUtils.testEncoding(
                                tsType, TSEncoding.CHIMP, columnFloatData.get(i), maxPoint, result, lossAcc);
                    }
                    totalOriginal += result[0];
                    totalEncoded += result[1];
                    totalCompTime += result[2];
                    totalDecompTime += result[3];
                    lossErr += lossAcc[0];
                    lossRef += lossAcc[1];
                } catch (Exception e) {
                    if (verbose) {
                        System.err.println("CHIMP128 patch failed");
                        e.printStackTrace(System.err);
                    }
                }
            }
            if (verbose && totalOriginal <= 0) {
                System.err.println("CHIMP128 produced zero original bytes for this patch");
            }
            totals.merge("CHIMP128", totalOriginal, totalEncoded, totalCompTime, totalDecompTime, lossErr, lossRef);
        }

        Processor.printColumnarEncodingRows64Accumulate(
                tsType,
                columnIntData,
                columnFloatData,
                columnFloatMaxPointData,
                totals,
                skipAlgorithms);
        System.setProperty(AlgorithmAllowList.INCREMENTAL_SKIP_ENV_KEY, AlgorithmAllowList.toEnvValue(downstreamSkipAlgorithms));
        ExternalCompressionBench.accumulateRows(
                tsType, columnIntData, columnFloatData, columnFloatMaxPointData, totals, downstreamSkipAlgorithms);
        ArrayList<ArrayList<Integer>> legacyInt =
                BenchmarkColumnBridge.toIntegerColumns(columnIntData);
        ArrayList<ArrayList<Float>> legacyFloat =
                BenchmarkColumnBridge.toFloatColumns(columnFloatData);
        ResearchNativeCliBench.accumulateRows64(
                tsType, columnIntData, columnFloatData, totals, downstreamSkipAlgorithms);
        ResearchFpcompressCliBench.accumulateRows64(
                tsType, columnIntData, columnFloatData, totals, downstreamSkipAlgorithms);
        ResearchSerfCliBench.accumulateRows(tsType, legacyInt, legacyFloat, totals, downstreamSkipAlgorithms);
        ResearchBuffCliBench.accumulateRows64(
                tsType, columnIntData, columnFloatData, totals, downstreamSkipAlgorithms);
        ResearchCmixCliBench.accumulateRows(tsType, legacyInt, legacyFloat, totals, downstreamSkipAlgorithms);
        ResearchZfpCliBench.accumulateRows64(
                tsType, columnIntData, columnFloatData, totals, downstreamSkipAlgorithms);

        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "ans")) {
            long[] acc = new long[4];
            double lossErr = 0;
            double lossRef = 0;
            double[] lossAcc = new double[2];
            for (byte[] data : columnByteData) {
                long[] result = new long[4];
                try {
                    java.util.Arrays.fill(lossAcc, 0);
                    CompressionUtils.testAnsEntropyCompression(data, result, lossAcc);
                    acc[0] += result[0];
                    acc[1] += result[1];
                    acc[2] += result[2];
                    acc[3] += result[3];
                    lossErr += lossAcc[0];
                    lossRef += lossAcc[1];
                } catch (Exception ignored) {
                    // skip
                }
            }
            totals.merge("ans", acc, lossErr, lossRef);
        }

        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "huffman")) {
            long[] acc = new long[4];
            double lossErr = 0;
            double lossRef = 0;
            double[] lossAcc = new double[2];
            for (byte[] data : columnByteData) {
                long[] result = new long[4];
                try {
                    java.util.Arrays.fill(lossAcc, 0);
                    CompressionUtils.testHuffmanCompression(data, result, lossAcc);
                    acc[0] += result[0];
                    acc[1] += result[1];
                    acc[2] += result[2];
                    acc[3] += result[3];
                    lossErr += lossAcc[0];
                    lossRef += lossAcc[1];
                } catch (Exception ignored) {
                    // skip
                }
            }
            totals.merge("huffman", acc, lossErr, lossRef);
        }

        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "arithmetic")) {
            long[] acc = new long[4];
            double lossErr = 0;
            double lossRef = 0;
            double[] lossAcc = new double[2];
            for (byte[] data : columnByteData) {
                long[] result = new long[4];
                try {
                    java.util.Arrays.fill(lossAcc, 0);
                    CompressionUtils.testArithmeticCompression(data, result, lossAcc);
                    acc[0] += result[0];
                    acc[1] += result[1];
                    acc[2] += result[2];
                    acc[3] += result[3];
                    lossErr += lossAcc[0];
                    lossRef += lossAcc[1];
                } catch (Exception ignored) {
                    // skip
                }
            }
            totals.merge("arithmetic", acc, lossErr, lossRef);
        }
    }

    private static void flushTotals(
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> existingAlgorithms,
            int columnIndex)
            throws Exception {
        for (Map.Entry<String, long[]> e : totals.snapshot().entrySet()) {
            if (AlgorithmAllowList.shouldSkip(existingAlgorithms, e.getKey())) {
                continue;
            }
            long[] t = e.getValue();
            if (t[0] <= 0) {
                continue;
            }
            csvWriter.printRecord(
                    columnIndex,
                    e.getKey(),
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    totals.isLossy(e.getKey()),
                    totals.lossRateCsv(e.getKey()));
            existingAlgorithms.add(e.getKey());
        }
    }

    private static String normalizeCell(String raw) {
        if (raw == null) {
            return "";
        }
        String t = raw.trim();
        if (t.isEmpty()) {
            return "";
        }
        if (t.startsWith("\"") && t.endsWith("\"") && t.length() >= 2) {
            t = t.substring(1, t.length() - 1).trim();
        }
        return ExperimentDataPolicy.isNaNCell(t) ? "" : t;
    }
}
