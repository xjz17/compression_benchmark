package org.example;

import java.io.*;

public class Main {

    /** Relative paths are resolved against {@code WEB_COMPRESSION_REPO_ROOT}, not {@code user.dir}. */
    static String normalizeIoPath(String path) {
        if (path == null || path.isEmpty()) {
            return path;
        }
        File f = new File(path);
        if (!f.isAbsolute()) {
            String repo = System.getenv("WEB_COMPRESSION_REPO_ROOT");
            if (repo != null && !repo.isBlank()) {
                f = new File(repo, path);
            }
        }
        return f.getAbsoluteFile().getPath();
    }

    public static void main(String[] args) throws Exception {
        String inputPath = null;
        String dataType = null;
        String outputReportPath = null;
        int lzmaLevel = 0;
        int zstdLevel = 3;
        String tsFileDir = null;
        int maxPointNumber = 2;
        boolean typedTsFileExport = false;
        String typedTsFileOutput = null;
        String typedTsFileInfoCsv = null;
        String typedTsFileIntEncoding = null;
        String typedTsFileFloatEncoding = null;
        String typedTsFileStringEncoding = null;
        String typedTsFileExportDir = null;
        String typedTsFileEncodingSummary = null;
        String columnTypesManifestPath = null;
        String columnInfoPath = null;
        String columnMaxPointsPath = null;
        Integer columnIndex = null;
        String runAlgorithms = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i":
                case "--input":
                    if (i + 1 < args.length) {
                        inputPath = args[++i];
                    } else {
                        System.err.println("Missing value for --input parameter");
                        return;
                    }
                    break;

                case "--data-type":
                    if (i + 1 < args.length) {
                        dataType = args[++i];
                    } else {
                        System.err.println("Missing value for --data-type parameter");
                        return;
                    }
                    break;

                case "--output-report":
                    if (i + 1 < args.length) {
                        outputReportPath = args[++i];
                    } else {
                        System.err.println("Missing value for --output-report parameter");
                        return;
                    }
                    break;

                case "--xz-level":
                case "--lzma-level":
                    if (i + 1 < args.length) {
                        try {
                            lzmaLevel = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid xz-level value: " + args[i]);
                            return;
                        }
                    } else {
                        System.err.println("Missing value for --xz-level parameter");
                        return;
                    }
                    break;

                case "--zstd-level":
                    if (i + 1 < args.length) {
                        try {
                            zstdLevel = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid zstd-level value: " + args[i]);
                            return;
                        }
                    } else {
                        System.err.println("Missing value for --zstd-level parameter");
                        return;
                    }
                    break;

                case "--tsfile-dir":
                    if (i + 1 < args.length) {
                        tsFileDir = args[++i];
                    } else {
                        System.err.println("Missing value for --tsfile-dir parameter");
                        return;
                    }
                    break;

                case "--max-point-number":
                    if (i + 1 < args.length) {
                        try {
                            maxPointNumber = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid max-point-number value: " + args[i]);
                        }
                    } else {
                        System.err.println("Missing value for --max-point-number parameter");
                    }
                    break;

                case "--typed-tsfile-export":
                    typedTsFileExport = true;
                    break;

                case "--typed-tsfile-output":
                    if (i + 1 < args.length) {
                        typedTsFileOutput = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-output parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-info-csv":
                    if (i + 1 < args.length) {
                        typedTsFileInfoCsv = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-info-csv parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-int-encoding":
                    if (i + 1 < args.length) {
                        typedTsFileIntEncoding = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-int-encoding parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-float-encoding":
                    if (i + 1 < args.length) {
                        typedTsFileFloatEncoding = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-float-encoding parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-string-encoding":
                    if (i + 1 < args.length) {
                        typedTsFileStringEncoding = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-string-encoding parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-export-dir":
                    if (i + 1 < args.length) {
                        typedTsFileExportDir = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-export-dir parameter");
                        return;
                    }
                    break;

                case "--typed-tsfile-encoding-summary":
                    if (i + 1 < args.length) {
                        typedTsFileEncodingSummary = args[++i];
                    } else {
                        System.err.println("Missing value for --typed-tsfile-encoding-summary parameter");
                        return;
                    }
                    break;

                case "--column-types-manifest":
                    if (i + 1 < args.length) {
                        columnTypesManifestPath = args[++i];
                    } else {
                        System.err.println("Missing value for --column-types-manifest parameter");
                        return;
                    }
                    break;

                case "--column-max-points":
                    if (i + 1 < args.length) {
                        columnMaxPointsPath = args[++i];
                    } else {
                        System.err.println("Missing value for --column-max-points parameter");
                        return;
                    }
                    break;

                case "--column-info":
                    if (i + 1 < args.length) {
                        columnInfoPath = args[++i];
                    } else {
                        System.err.println("Missing value for --column-info parameter");
                        return;
                    }
                    break;

                case "--column-index":
                    if (i + 1 < args.length) {
                        try {
                            columnIndex = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid column-index value: " + args[i]);
                            return;
                        }
                    } else {
                        System.err.println("Missing value for --column-index parameter");
                        return;
                    }
                    break;

                case "--algorithms":
                    if (i + 1 < args.length) {
                        runAlgorithms = args[++i];
                    } else {
                        System.err.println("Missing value for --algorithms parameter");
                        return;
                    }
                    break;

                default:
                    System.err.println("Unknown parameter: " + args[i] + ", ignoring");
                    break;
            }
        }

        if (typedTsFileExport) {
            String manifest = columnInfoPath != null ? columnInfoPath : columnTypesManifestPath;
            if (inputPath == null || manifest == null) {
                System.err.println(
                        "Typed TsFile export requires: --input <csv> and --column-info <csv> (or legacy --column-types-manifest)");
                return;
            }
            inputPath = normalizeIoPath(inputPath);
            manifest = normalizeIoPath(manifest);
            if (typedTsFileOutput == null || typedTsFileInfoCsv == null) {
                System.err.println(
                        "Typed TsFile export requires: --typed-tsfile-output and --typed-tsfile-info-csv");
                return;
            }
            if (typedTsFileIntEncoding == null
                    || typedTsFileFloatEncoding == null
                    || typedTsFileStringEncoding == null) {
                System.err.println(
                        "Typed TsFile export requires: --typed-tsfile-int-encoding, --typed-tsfile-float-encoding, --typed-tsfile-string-encoding");
                return;
            }
            typedTsFileOutput = normalizeIoPath(typedTsFileOutput);
            typedTsFileInfoCsv = normalizeIoPath(typedTsFileInfoCsv);
            Processor.exportTypedCsvToSelectedTsFile(
                    inputPath,
                    manifest,
                    typedTsFileOutput,
                    typedTsFileInfoCsv,
                    typedTsFileIntEncoding,
                    typedTsFileFloatEncoding,
                    typedTsFileStringEncoding);
            return;
        }

        if (inputPath == null || dataType == null || outputReportPath == null) {
            System.err.println("Missing required parameters");
            return;
        }

        inputPath = normalizeIoPath(inputPath);
        outputReportPath = normalizeIoPath(outputReportPath);
        if (tsFileDir != null) {
            tsFileDir = normalizeIoPath(tsFileDir);
        }
        if (columnInfoPath != null) {
            columnInfoPath = normalizeIoPath(columnInfoPath);
        }
        if (columnMaxPointsPath != null) {
            columnMaxPointsPath = normalizeIoPath(columnMaxPointsPath);
        }
        if (runAlgorithms != null && !runAlgorithms.isBlank()) {
            System.setProperty(AlgorithmAllowList.ALLOWLIST_ENV_KEY, runAlgorithms.trim());
        }

        File outputFile = new File(outputReportPath);
        if (!outputFile.getParentFile().exists()) {
            outputFile.getParentFile().mkdirs();
        }

        if (inputPath.toLowerCase().endsWith(".csv")) {
            if (tsFileDir != null) {
                // TsFile page / regenerate_tsfile_staticdata.py: one CSV pass per algorithm, stream rows.
                Processor.processCsvSimple(
                        inputPath,
                        dataType,
                        maxPointNumber,
                        columnInfoPath,
                        columnMaxPointsPath,
                        tsFileDir,
                        outputReportPath);
            } else {
                // Home / catalog staticdata (patch rows + column_info max points).
                Processor.processCsvSimple(inputPath, dataType, outputReportPath, columnInfoPath, columnIndex);
            }
        }
    }
}
