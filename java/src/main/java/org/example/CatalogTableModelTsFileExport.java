package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.tsfile.common.conf.TSFileDescriptor;
import org.apache.tsfile.enums.ColumnCategory;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.TableSchema;
import org.apache.tsfile.file.metadata.enums.CompressionType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.read.v4.ITsFileReader;
import org.apache.tsfile.read.v4.TsFileReaderBuilder;
import org.apache.tsfile.write.record.Tablet;
import org.apache.tsfile.write.schema.IMeasurementSchema;
import org.apache.tsfile.write.schema.MeasurementSchema;
import org.apache.tsfile.write.v4.ITsFileWriter;
import org.apache.tsfile.write.v4.TsFileWriterBuilder;
import org.apache.tsfile.encoding.encoder.Encoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Catalog CSV → table-model TsFile (TsFile ≥ 2.3.x) with fixed encodings and LZ4 compression. */
final class CatalogTableModelTsFileExport {

    static final String TSFILE_MODEL = "table";
    static final int TAG_COLUMN_COUNT = 0;
    static final String INT_ENCODING = "TS_2DIFF";
    static final String FLOAT_ENCODING = "GORILLA";
    static final String STRING_ENCODING = "PLAIN";
    static final String TIME_ENCODING = "TS_2DIFF";
    static final String COMPRESSION = "LZ4";

    private static final int TABLET_ROW_BATCH = 10_000;

    private CatalogTableModelTsFileExport() {}

    static final class ColumnSpec {
        final int index;
        final String name;
        final String logicalType;
        final int maxPointNumber;

        ColumnSpec(int index, String name, String logicalType, int maxPointNumber) {
            this.index = index;
            this.name = name;
            this.logicalType = logicalType;
            this.maxPointNumber = maxPointNumber;
        }
    }

    static void export(
            String csvPath,
            String columnInfoPath,
            String outputTsFilePath,
            String tableName)
            throws Exception {
        File csvFile = new File(csvPath);
        if (!csvFile.isFile()) {
            throw new FileNotFoundException("CSV not found: " + csvPath);
        }
        File manifestFile = new File(columnInfoPath);
        if (!manifestFile.isFile()) {
            throw new FileNotFoundException("Column info not found: " + columnInfoPath);
        }

        String[] headers;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile, StandardCharsets.UTF_8))) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("CSV is empty");
            }
            headers = headerLine.split(",", -1);
        }

        List<ColumnSpec> columns = readColumnSpecs(columnInfoPath);
        if (columns.isEmpty()) {
            throw new IOException("No exportable columns in column_info");
        }

        String resolvedTableName = sanitizeTableName(tableName, outputTsFilePath);
        configureTsFileDefaults();

        Map<Integer, String> measurementByIndex = new LinkedHashMap<>();
        TableSchema tableSchema = buildTableSchema(resolvedTableName, columns, headers, measurementByIndex);
        List<String> tabletColumnNames = tabletColumnNames(tableSchema);

        File tsFile = new File(outputTsFilePath);
        if (tsFile.getParentFile() != null) {
            tsFile.getParentFile().mkdirs();
        }
        if (tsFile.isFile()) {
            Files.delete(tsFile.toPath());
        }

        long rowCount = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile, StandardCharsets.UTF_8));
                ITsFileWriter writer =
                        new TsFileWriterBuilder()
                                .file(tsFile)
                                .tableSchema(tableSchema)
                                .build()) {
            if (br.readLine() == null) {
                throw new IOException("CSV is empty");
            }

            Tablet tablet =
                    new Tablet(
                            resolvedTableName,
                            tabletColumnNames,
                            IMeasurementSchema.getDataTypeList(tableSchema.getColumnSchemas()),
                            tableSchema.getColumnTypes(),
                            TABLET_ROW_BATCH);

            String line;
            long timestamp = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",", -1);
                int rowIdx = tablet.getRowSize();
                tablet.addTimestamp(rowIdx, timestamp++);
                appendRowValues(tablet, rowIdx, columns, values, measurementByIndex);
                rowCount++;
                if (tablet.getRowSize() >= TABLET_ROW_BATCH) {
                    writer.write(tablet);
                    tablet.reset();
                }
            }
            if (tablet.getRowSize() > 0) {
                writer.write(tablet);
            }
        }

        verifyTableModelTsFile(tsFile, resolvedTableName, rowCount);
        System.out.println(
                "Catalog table-model TsFile export: "
                        + tsFile.getAbsolutePath()
                        + " table="
                        + resolvedTableName
                        + " rows="
                        + rowCount
                        + " model="
                        + TSFILE_MODEL
                        + " compression="
                        + COMPRESSION);
    }

    private static void configureTsFileDefaults() {
        TSFileDescriptor.getInstance().getConfig().setTimeEncoder(TIME_ENCODING);
        TSFileDescriptor.getInstance().getConfig().setCompressor(COMPRESSION);
    }

    private static TableSchema buildTableSchema(
            String tableName,
            List<ColumnSpec> columns,
            String[] headers,
            Map<Integer, String> measurementByIndex)
            throws IOException {
        List<IMeasurementSchema> schemas = new ArrayList<>();
        List<ColumnCategory> categories = new ArrayList<>();
        Set<String> usedNames = new HashSet<>();

        for (ColumnSpec col : columns) {
            if (col.index < 0 || col.index >= headers.length) {
                throw new IOException("column_index out of range: " + col.index);
            }
            String baseName = col.name != null && !col.name.isBlank() ? col.name : headers[col.index];
            String measurementName = uniqueMeasurementName(baseName, col.index, usedNames);
            measurementByIndex.put(col.index, measurementName);
            switch (col.logicalType) {
                case "integer":
                    schemas.add(
                            new MeasurementSchema(
                                    measurementName,
                                    TSDataType.INT64,
                                    TSEncoding.TS_2DIFF,
                                    CompressionType.LZ4));
                    categories.add(ColumnCategory.FIELD);
                    break;
                case "float":
                    Map<String, String> props = new HashMap<>();
                    props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(Math.max(0, col.maxPointNumber)));
                    schemas.add(
                            new MeasurementSchema(
                                    measurementName,
                                    TSDataType.DOUBLE,
                                    TSEncoding.GORILLA,
                                    CompressionType.LZ4,
                                    props));
                    categories.add(ColumnCategory.FIELD);
                    break;
                case "string":
                    schemas.add(
                            new MeasurementSchema(
                                    measurementName,
                                    TSDataType.STRING,
                                    TSEncoding.PLAIN,
                                    CompressionType.LZ4));
                    categories.add(ColumnCategory.FIELD);
                    break;
                default:
                    throw new IOException("Unsupported logical_type: " + col.logicalType);
            }
        }
        return new TableSchema(tableName, schemas, categories);
    }

    private static List<String> tabletColumnNames(TableSchema tableSchema) {
        return IMeasurementSchema.getMeasurementNameList(tableSchema.getColumnSchemas());
    }

    private static void appendRowValues(
            Tablet tablet,
            int rowIdx,
            List<ColumnSpec> columns,
            String[] values,
            Map<Integer, String> measurementByIndex) {
        for (ColumnSpec col : columns) {
            if (col.index >= values.length) {
                continue;
            }
            String name = measurementByIndex.get(col.index);
            if (name == null) {
                continue;
            }
            String raw = values[col.index];
            switch (col.logicalType) {
                case "integer":
                    {
                        String cell = Processor.normalizeCsvNumericCell(raw);
                        if (cell.isEmpty()) {
                            continue;
                        }
                        try {
                            tablet.addValue(rowIdx, name, Long.parseLong(cell));
                        } catch (NumberFormatException ignored) {
                            // skip
                        }
                    }
                    break;
                case "float":
                    {
                        String cell = Processor.normalizeCsvNumericCell(raw);
                        if (cell.isEmpty()) {
                            continue;
                        }
                        try {
                            double value = Double.parseDouble(cell);
                            if (!ExperimentDataPolicy.keepNumericValue(value)) continue;
                            tablet.addValue(rowIdx, name, value);
                        } catch (NumberFormatException ignored) {
                            // skip
                        }
                    }
                    break;
                case "string":
                    if (raw == null) {
                        continue;
                    }
                    tablet.addValue(rowIdx, name, raw);
                    break;
                default:
                    break;
            }
        }
    }

    private static List<ColumnSpec> readColumnSpecs(String columnInfoPath) throws IOException {
        List<ColumnSpec> out = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(java.nio.file.Paths.get(columnInfoPath), StandardCharsets.UTF_8)) {
            String header = br.readLine();
            if (header == null) {
                throw new IOException("Column info is empty");
            }
            StringBuilder csvChunk = new StringBuilder();
            csvChunk.append(header).append('\n');
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
                    String logical = rec.get("logical_type").trim().toLowerCase(Locale.ROOT);
                    int mp = 0;
                    try {
                        mp = Integer.parseInt(rec.get("max_point_number"));
                    } catch (NumberFormatException ignored) {
                        // 0
                    }
                    if (!logical.equals("integer") && !logical.equals("float") && !logical.equals("string")) {
                        continue;
                    }
                    out.add(new ColumnSpec(idx, name, logical, mp));
                }
            }
        }
        out.sort((a, b) -> Integer.compare(a.index, b.index));
        return out;
    }

    static String sanitizeTableNameForInfo(String tableName) {
        return sanitizeTableName(tableName, tableName + ".tsfile");
    }

    private static String sanitizeTableName(String tableName, String outputTsFilePath) {
        String candidate = tableName;
        if (candidate == null || candidate.isBlank()) {
            String base = new File(outputTsFilePath).getName();
            int dot = base.lastIndexOf('.');
            candidate = dot > 0 ? base.substring(0, dot) : base;
        }
        String sanitized = candidate.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_]+", "_");
        if (sanitized.isBlank()) {
            sanitized = "catalog";
        }
        return sanitized;
    }

    private static String uniqueMeasurementName(String rawName, int columnIndex, Set<String> usedNames) {
        String base = rawName == null ? "" : rawName.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_]+", "_");
        if (base.isBlank()) {
            base = "col_" + columnIndex;
        }
        String candidate = base;
        int suffix = 1;
        while (!usedNames.add(candidate)) {
            candidate = base + "_" + suffix++;
        }
        return candidate;
    }

    private static void verifyTableModelTsFile(File tsFile, String tableName, long expectedRows)
            throws IOException {
        if (!tsFile.isFile() || tsFile.length() <= 0) {
            throw new IOException("TsFile was not created or is empty: " + tsFile.getAbsolutePath());
        }
        try (ITsFileReader reader = new TsFileReaderBuilder().file(tsFile).build()) {
            if (reader.getTableSchemas(tableName).isEmpty()) {
                throw new IOException("Table schema missing in TsFile: " + tableName);
            }
            if (reader.getTableSchemas(tableName).get().getColumnSchemas().isEmpty()) {
                throw new IOException("Table has no columns: " + tableName);
            }
            if (reader.getTableSchemas(tableName).get().getTagColumnCnt() != TAG_COLUMN_COUNT) {
                throw new IOException(
                        "Unexpected TAG column count in TsFile: "
                                + reader.getTableSchemas(tableName).get().getTagColumnCnt());
            }
        }
        if (expectedRows <= 0) {
            throw new IOException("No data rows written to TsFile");
        }
    }
}
