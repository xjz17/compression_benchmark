package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.CompressionType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.read.TsFileReader;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.query.dataset.QueryDataSet;
import org.apache.tsfile.write.TsFileWriter;
import org.apache.tsfile.write.record.TSRecord;
import org.apache.tsfile.utils.Binary;
import org.apache.tsfile.write.record.datapoint.DoubleDataPoint;
import org.apache.tsfile.write.record.datapoint.LongDataPoint;
import org.apache.tsfile.write.record.datapoint.StringDataPoint;
import org.apache.tsfile.write.schema.IMeasurementSchema;
import org.apache.tsfile.write.schema.MeasurementSchema;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TsFile staticdata bench ({@code regenerate_tsfile_staticdata.py}): for each chunk compressor / TsFile
 * encoding, read the CSV once and stream rows into a TsFile (no full-dataset materialization).
 * Per-column {@code max_point_number} comes from {@code {dataset}_column_info.csv}.
 */
final class ProcessorTsfileStaticdataBench {
    private static final int DEFAULT_PATCH_ROWS = 10_000;
    private static final String DEVICE_ID = "device_1";

    private ProcessorTsfileStaticdataBench() {}

    static void run(
            String csvPath,
            String dataType,
            int maxPointNumberFallback,
            String columnInfoPath,
            String columnMaxPointsPath,
            String tsfileDir,
            String outputReportPath) {
        File tmpReport = null;
        boolean reportReplaced = false;
        try {
            tmpReport = Processor.createAtomicReportTempFile(outputReportPath);
            TsfilePlan plan =
                    buildPlan(
                            csvPath,
                            dataType,
                            maxPointNumberFallback,
                            columnInfoPath,
                            columnMaxPointsPath);
            File outputDir = new File(tsfileDir);
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }

            try (CSVPrinter csvWriter =
                    new CSVPrinter(
                            new FileWriter(tmpReport, StandardCharsets.UTF_8),
                            CSVFormat.DEFAULT
                                    .withHeader(
                                            "algorithm",
                                            "originalSize",
                                            "compressedSize",
                                            "compressionTime",
                                            "decompressionTime")
                                    .withRecordSeparator("\n"))) {

                List<CompressionType> compressionList = new ArrayList<>();
                compressionList.add(CompressionType.GZIP);
                compressionList.add(CompressionType.LZ4);
                compressionList.add(CompressionType.LZMA2);
                compressionList.add(CompressionType.SNAPPY);
                compressionList.add(CompressionType.ZSTD);

                for (CompressionType compression : compressionList) {
                    String tsFileName =
                            plan.baseName
                                    + "_"
                                    + tsfileChunkCompressionAlgorithmKey(compression)
                                    + ".tsfile";
                    String tsFilePath = new File(outputDir, tsFileName).getAbsolutePath();
                    System.out.println("Processing compression: " + compression + " with PLAIN encoding");
                    try {
                        long[] metrics =
                                streamCsvToTsfile(
                                        csvPath,
                                        plan,
                                        tsFilePath,
                                        TSEncoding.PLAIN,
                                        compression);
                        csvWriter.printRecord(
                                tsfileChunkCompressionAlgorithmKey(compression),
                                metrics[0],
                                metrics[1],
                                metrics[2],
                                metrics[3]);
                    } catch (Exception e) {
                        System.err.println("Error processing compression " + compression + ": " + e.getMessage());
                        e.printStackTrace();
                    }
                }

                List<TSEncoding> encodingList;
                if (plan.tsType == TSDataType.STRING) {
                    encodingList = supportedEncodingsFor(TSDataType.STRING);
                } else {
                    encodingList = new ArrayList<>();
                    encodingList.add(TSEncoding.CHIMP);
                    encodingList.add(TSEncoding.GORILLA);
                    encodingList.add(TSEncoding.RLE);
                    encodingList.add(TSEncoding.RLBE);
                    encodingList.add(TSEncoding.SPRINTZ);
                    encodingList.add(TSEncoding.TS_2DIFF);
                }

                for (TSEncoding encoding : encodingList) {
                    System.out.println("Processing encoding: " + encoding);
                    String tsFileName = plan.baseName + "_" + encoding + ".tsfile";
                    String tsFilePath = new File(outputDir, tsFileName).getAbsolutePath();
                    try {
                        long[] metrics =
                                streamCsvToTsfile(
                                        csvPath,
                                        plan,
                                        tsFilePath,
                                        encoding,
                                        CompressionType.UNCOMPRESSED);
                        csvWriter.printRecord(
                                encoding.toString(), metrics[0], metrics[1], metrics[2], metrics[3]);
                    } catch (Exception e) {
                        System.err.println("Error processing encoding " + encoding + ": " + e.getMessage());
                        e.printStackTrace();
                    }
                }

                if (plan.tsType != TSDataType.STRING) {
                    System.out.println("Processing TS_2DIFF+BOS (research)");
                    writeTs2DiffBosRow(csvPath, plan, outputDir, csvWriter);
                }

                csvWriter.flush();
            }
            Processor.replaceReportFileAtomically(tmpReport, outputReportPath);
            reportReplaced = true;
            System.out.println("TSFile compression results written to " + outputReportPath);
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

    private static void writeTs2DiffBosRow(
            String csvPath,
            TsfilePlan plan,
            File outputDir,
            CSVPrinter csvWriter)
            throws Exception {
        String bosKey = "ts_2diff_bos";
        String tsFilePath = new File(outputDir, plan.baseName + "_" + bosKey + ".tsfile").getAbsolutePath();

        long bosEncodeStart = System.nanoTime();
        ColumnSeries series = loadColumnSeriesInPatches(csvPath, plan);
        List<byte[]> bosPayloads =
                plan.tsType == TSDataType.INT64
                        ? Ts2DiffBosImprove.encodeLongColumnBosPayloadsOrNull(series.longCols)
                        : Ts2DiffBosImprove.encodeDoubleColumnBosPayloadsOrNull(series.doubleCols);

        if (bosPayloads == null || bosPayloads.isEmpty()) {
            return;
        }
        long bosPayloadBytes = 0L;
        for (byte[] p : bosPayloads) {
            bosPayloadBytes += p.length;
        }

        try {
            TSEncoding encoding = TSEncoding.TS_2DIFF;
            try (TsFileWriter writer = new TsFileWriter(new File(tsFilePath))) {
                Path devicePath = new Path(DEVICE_ID);
                List<IMeasurementSchema> measurementSchemas =
                        buildMeasurementSchemas(
                                plan, encoding, CompressionType.UNCOMPRESSED);
                if (measurementSchemas.isEmpty()) {
                    return;
                }
                writer.registerTimeseries(devicePath, measurementSchemas);
                writeSeriesToTsfile(writer, plan, series);
            }
            long writeEnd = System.nanoTime();
            long compressionTime = writeEnd - bosEncodeStart;
            long decompressionTime = measureTsfileReadNanos(tsFilePath, plan);
            csvWriter.printRecord(
                    bosKey, series.totalOriginalBytes, bosPayloadBytes, compressionTime, decompressionTime);
        } catch (Exception e) {
            System.err.println("Error processing TS_2DIFF+BOS: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /** One CSV read per TsFile; stream each row into the writer. Returns orig, compressed, comp ns, decomp ns. */
    private static long[] streamCsvToTsfile(
            String csvPath,
            TsfilePlan plan,
            String tsFilePath,
            TSEncoding encoding,
            CompressionType compression)
            throws Exception {
        long startTime = System.nanoTime();
        long totalOriginal = 0L;
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath, StandardCharsets.UTF_8));
                TsFileWriter writer = new TsFileWriter(new File(tsFilePath))) {
            if (br.readLine() == null) {
                throw new IOException("CSV is empty");
            }
            List<IMeasurementSchema> measurementSchemas =
                    buildMeasurementSchemas(plan, encoding, compression);
            if (measurementSchemas.isEmpty()) {
                return new long[] {0, 0, 0, 0};
            }
            writer.registerTimeseries(new Path(DEVICE_ID), measurementSchemas);

            String line;
            long rowIndex = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                TSRecord record = new TSRecord(DEVICE_ID, rowIndex++);
                int tuples = 0;
                for (ActiveColumn ac : plan.activeColumns) {
                    if (ac.headerIndex >= values.length) {
                        continue;
                    }
                    if (plan.tsType == TSDataType.STRING) {
                        String raw = values[ac.headerIndex];
                        if (raw == null || raw.isEmpty()) {
                            continue;
                        }
                        byte[] utf8 = raw.getBytes(StandardCharsets.UTF_8);
                        record.addTuple(new StringDataPoint(ac.measurementName, new Binary(utf8)));
                        totalOriginal += utf8.length;
                        tuples++;
                        continue;
                    }
                    String cell = Processor.normalizeCsvNumericCell(values[ac.headerIndex]);
                    if (cell.isEmpty()) {
                        continue;
                    }
                    try {
                        if (plan.tsType == TSDataType.INT64) {
                            long v = Long.parseLong(cell);
                            record.addTuple(new LongDataPoint(ac.measurementName, v));
                            totalOriginal += BenchmarkWireFormat.INT_BYTES;
                            tuples++;
                        } else {
                            double v = Double.parseDouble(cell);
                            if (!ExperimentDataPolicy.keepNumericValue(v)) continue;
                            record.addTuple(new DoubleDataPoint(ac.measurementName, v));
                            totalOriginal += BenchmarkWireFormat.FLOAT_BYTES;
                            tuples++;
                        }
                    } catch (NumberFormatException ignored) {
                        // skip cell
                    }
                }
                if (tuples > 0) {
                    writer.writeRecord(record);
                }
            }
        }
        long compressionTime = System.nanoTime() - startTime;
        File tsFile = new File(tsFilePath);
        long compressedSize = tsFile.exists() ? tsFile.length() : 0;
        long decompressionTime = measureTsfileReadNanos(tsFilePath, plan);
        return new long[] {totalOriginal, compressedSize, compressionTime, decompressionTime};
    }

    private static long measureTsfileReadNanos(String tsFilePath, TsfilePlan plan) {
        try {
            long decompStart = System.nanoTime();
            try (TsFileSequenceReader fileReader = new TsFileSequenceReader(tsFilePath);
                    TsFileReader tsFileReader = new TsFileReader(fileReader)) {
                ArrayList<Path> queryPaths = new ArrayList<>();
                for (ActiveColumn ac : plan.activeColumns) {
                    queryPaths.add(new Path(DEVICE_ID, ac.measurementName, true));
                }
                if (!queryPaths.isEmpty()) {
                    QueryExpression queryExpression = QueryExpression.create(queryPaths, null);
                    QueryDataSet queryDataSet = tsFileReader.query(queryExpression);
                    while (queryDataSet.hasNext()) {
                        RowRecord record = queryDataSet.next();
                        if (record == null) {
                            break;
                        }
                    }
                }
            }
            return System.nanoTime() - decompStart;
        } catch (Exception e) {
            System.err.println("Error reading TSFile for decompression measurement: " + e.getMessage());
            return 0L;
        }
    }

    private static ColumnSeries loadColumnSeriesInPatches(String csvPath, TsfilePlan plan) throws IOException {
        int n = plan.activeColumns.size();
        ArrayList<ArrayList<Long>> longCols = new ArrayList<>();
        ArrayList<ArrayList<Double>> doubleCols = new ArrayList<>();
        if (plan.tsType == TSDataType.INT64) {
            for (int i = 0; i < n; i++) {
                longCols.add(new ArrayList<>());
            }
        } else {
            for (int i = 0; i < n; i++) {
                doubleCols.add(new ArrayList<>());
            }
        }
        long totalOriginal = 0L;
        int patchLimit = patchRowLimit();
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath, StandardCharsets.UTF_8))) {
            if (br.readLine() == null) {
                return new ColumnSeries(longCols, doubleCols, 0L);
            }
            String line;
            int rowsInPatch = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                for (int ci = 0; ci < plan.activeColumns.size(); ci++) {
                    ActiveColumn ac = plan.activeColumns.get(ci);
                    if (ac.headerIndex >= values.length) {
                        continue;
                    }
                    String cell = Processor.normalizeCsvNumericCell(values[ac.headerIndex]);
                    if (cell.isEmpty()) {
                        continue;
                    }
                    try {
                        if (plan.tsType == TSDataType.INT64) {
                            long v = Long.parseLong(cell);
                            longCols.get(ci).add(v);
                            totalOriginal += BenchmarkWireFormat.INT_BYTES;
                        } else {
                            double v = Double.parseDouble(cell);
                            if (!ExperimentDataPolicy.keepNumericValue(v)) continue;
                            doubleCols.get(ci).add(v);
                            totalOriginal += BenchmarkWireFormat.FLOAT_BYTES;
                        }
                    } catch (NumberFormatException ignored) {
                        // skip
                    }
                }
                rowsInPatch++;
                if (rowsInPatch >= patchLimit) {
                    rowsInPatch = 0;
                }
            }
        }
        return new ColumnSeries(longCols, doubleCols, totalOriginal);
    }

    private static void writeSeriesToTsfile(TsFileWriter writer, TsfilePlan plan, ColumnSeries series)
            throws Exception {
        int maxRows =
                plan.tsType == TSDataType.INT64
                        ? series.longCols.stream().mapToInt(List::size).max().orElse(0)
                        : series.doubleCols.stream().mapToInt(List::size).max().orElse(0);
        for (int rowIndex = 0; rowIndex < maxRows; rowIndex++) {
            TSRecord record = new TSRecord(DEVICE_ID, rowIndex);
            for (int ci = 0; ci < plan.activeColumns.size(); ci++) {
                ActiveColumn ac = plan.activeColumns.get(ci);
                if (plan.tsType == TSDataType.INT64) {
                    ArrayList<Long> col = series.longCols.get(ci);
                    if (rowIndex < col.size()) {
                        record.addTuple(new LongDataPoint(ac.measurementName, col.get(rowIndex)));
                    }
                } else {
                    ArrayList<Double> col = series.doubleCols.get(ci);
                    if (rowIndex < col.size()) {
                        record.addTuple(new DoubleDataPoint(ac.measurementName, col.get(rowIndex)));
                    }
                }
            }
            writer.writeRecord(record);
        }
    }

    private static List<IMeasurementSchema> buildMeasurementSchemas(
            TsfilePlan plan, TSEncoding encoding, CompressionType compression) {
        List<IMeasurementSchema> measurementSchemas = new ArrayList<>();
        for (ActiveColumn ac : plan.activeColumns) {
            Map<String, String> props = new HashMap<>();
            props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(ac.maxPointNumber));
            measurementSchemas.add(
                    new MeasurementSchema(
                            ac.measurementName, plan.tsType, encoding, compression, props));
        }
        return measurementSchemas;
    }

    private static TsfilePlan buildPlan(
            String csvPath,
            String dataType,
            int maxPointNumberFallback,
            String columnInfoPath,
            String columnMaxPointsPath)
            throws IOException {
        TSDataType tsType;
        if ("int".equalsIgnoreCase(dataType)) {
            tsType = TSDataType.INT64;
        } else if ("float".equalsIgnoreCase(dataType)) {
            tsType = TSDataType.DOUBLE;
        } else if ("string".equalsIgnoreCase(dataType)) {
            tsType = TSDataType.STRING;
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }

        String[] headers;
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath, StandardCharsets.UTF_8))) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("CSV is empty");
            }
            headers = headerLine.split(",", -1);
        }

        Map<String, ColumnInfoRow> infoByName = loadColumnInfoByName(columnInfoPath);
        int[] maxPointsByHeaderIndex =
                readColumnMaxPointsByHeaderIndex(
                        columnMaxPointsPath, headers.length, maxPointNumberFallback, infoByName, headers);

        List<ActiveColumn> active = new ArrayList<>();
        if (!infoByName.isEmpty()) {
            for (int i = 0; i < headers.length; i++) {
                String name = headers[i].trim();
                ColumnInfoRow row = infoByName.get(name);
                if (row == null) {
                    continue;
                }
                String logical = row.logicalType == null ? "" : row.logicalType.trim();
                boolean include;
                if ("int".equalsIgnoreCase(dataType)) {
                    include = "integer".equalsIgnoreCase(logical);
                } else if ("float".equalsIgnoreCase(dataType)) {
                    include = "float".equalsIgnoreCase(logical);
                } else {
                    include = "string".equalsIgnoreCase(logical);
                }
                if (!include) {
                    continue;
                }
                int mp = tsType == TSDataType.DOUBLE ? maxPointsByHeaderIndex[i] : 0;
                active.add(new ActiveColumn(i, "index_" + i, mp));
            }
        } else {
            for (int i = 0; i < headers.length; i++) {
                int mp = tsType == TSDataType.DOUBLE ? maxPointsByHeaderIndex[i] : 0;
                active.add(new ActiveColumn(i, "index_" + i, mp));
            }
        }

        String csvFileName = new File(csvPath).getName();
        String baseName = csvFileName.substring(0, csvFileName.lastIndexOf('.'));
        return new TsfilePlan(baseName, headers, tsType, active);
    }

    private static Map<String, ColumnInfoRow> loadColumnInfoByName(String columnInfoPath) throws IOException {
        Map<String, ColumnInfoRow> out = new HashMap<>();
        if (columnInfoPath == null || columnInfoPath.isBlank()) {
            return out;
        }
        java.nio.file.Path path = java.nio.file.Paths.get(columnInfoPath);
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
                String name = rec.get("column_name");
                if (name == null) {
                    continue;
                }
                int mp = 0;
                try {
                    mp = Integer.parseInt(rec.get("max_point_number"));
                } catch (NumberFormatException ignored) {
                    // keep 0
                }
                out.put(
                        name.trim(),
                        new ColumnInfoRow(
                                rec.get("logical_type") == null ? "" : rec.get("logical_type").trim(),
                                Math.max(0, mp)));
            }
        }
        return out;
    }

    private static int[] readColumnMaxPointsByHeaderIndex(
            String columnMaxPointsPath,
            int colCount,
            int fallback,
            Map<String, ColumnInfoRow> infoByName,
            String[] headers)
            throws IOException {
        int[] out = new int[colCount];
        Arrays.fill(out, Math.max(0, fallback));
        for (int i = 0; i < headers.length; i++) {
            ColumnInfoRow row = infoByName.get(headers[i].trim());
            if (row != null) {
                out[i] = row.maxPointNumber;
            }
        }
        if (columnMaxPointsPath != null && !columnMaxPointsPath.isBlank() && infoByName.isEmpty()) {
            java.nio.file.Path path = java.nio.file.Paths.get(columnMaxPointsPath);
            if (Files.isRegularFile(path)) {
                try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
                    String line = br.readLine();
                    if (line != null) {
                        StringBuilder csvChunk = new StringBuilder();
                        csvChunk.append(line).append('\n');
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
                                                        "max_point_number")
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
                }
            }
        }
        return out;
    }

    private static int patchRowLimit() {
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

    private static final class ColumnInfoRow {
        final String logicalType;
        final int maxPointNumber;

        ColumnInfoRow(String logicalType, int maxPointNumber) {
            this.logicalType = logicalType;
            this.maxPointNumber = maxPointNumber;
        }
    }

    private static final class ActiveColumn {
        final int headerIndex;
        final String measurementName;
        final int maxPointNumber;

        ActiveColumn(int headerIndex, String measurementName, int maxPointNumber) {
            this.headerIndex = headerIndex;
            this.measurementName = measurementName;
            this.maxPointNumber = maxPointNumber;
        }
    }

    private static final class TsfilePlan {
        final String baseName;
        final String[] headers;
        final TSDataType tsType;
        final List<ActiveColumn> activeColumns;

        TsfilePlan(String baseName, String[] headers, TSDataType tsType, List<ActiveColumn> activeColumns) {
            this.baseName = baseName;
            this.headers = headers;
            this.tsType = tsType;
            this.activeColumns = activeColumns;
        }
    }

    private static final class ColumnSeries {
        final ArrayList<ArrayList<Long>> longCols;
        final ArrayList<ArrayList<Double>> doubleCols;
        final long totalOriginalBytes;

        ColumnSeries(
                ArrayList<ArrayList<Long>> longCols,
                ArrayList<ArrayList<Double>> doubleCols,
                long totalOriginalBytes) {
            this.longCols = longCols;
            this.doubleCols = doubleCols;
            this.totalOriginalBytes = totalOriginalBytes;
        }
    }
}
