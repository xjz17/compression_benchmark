package org.example;

import org.apache.commons.csv.CSVPrinter;
import org.apache.tsfile.enums.TSDataType;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Native CLIs for error-bounded (lossy) research codecs when binaries exist.
 * SZ float: {@code external/compression/SZ/example/sz} or CMake {@code SZ/build/bin/sz} + {@code sz.config},
 * or {@code WEB_COMPRESSION_SZ}.
 */
public final class ResearchNativeCliBench {

    private ResearchNativeCliBench() {
    }

    private static final class SzFloatPreset {
        final String rowName;
        final String errMode;
        final String boundFlag;
        final String boundValue;

        SzFloatPreset(String rowName, String errMode, String boundFlag, String boundValue) {
            this.rowName = rowName;
            this.errMode = errMode;
            this.boundFlag = boundFlag;
            this.boundValue = boundValue;
        }
    }

    private static final SzFloatPreset[] SZ_FLOAT_PRESETS = {
            new SzFloatPreset("sz_ABS_1e-1", "ABS", "-A", "1E-1"),
            new SzFloatPreset("sz_ABS_1e-2", "ABS", "-A", "1E-2"),
            new SzFloatPreset("sz_ABS_1e-3", "ABS", "-A", "1E-3"),
            new SzFloatPreset("sz_ABS_1e-4", "ABS", "-A", "1E-4"),
            new SzFloatPreset("sz_REL_1e-2", "REL", "-R", "1E-2"),
            new SzFloatPreset("sz_REL_1e-3", "REL", "-R", "1E-3"),
    };

    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runSzRows(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runSzRows(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    public static void accumulateRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runSzRows64(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runSzRows64(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    private static void runSzRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        ArrayList<ArrayList<Float>> columns =
                BenchmarkColumns.floatColumnsForCodecBench(tsType, intCols, floatCols);
        if (!columns.isEmpty()) {
            // SZ / SZ3: on by default when binary exists; set WEB_COMPRESSION_{SZ,SZ3}_BENCH=0 to skip.
            final String szFlag = System.getenv("WEB_COMPRESSION_SZ_BENCH");
            final boolean szBench = szFlag == null || szFlag.isEmpty() || "1".equals(szFlag.trim());
            final String sz3Flag = System.getenv("WEB_COMPRESSION_SZ3_BENCH");
            final boolean sz3Bench = sz3Flag == null || sz3Flag.isEmpty() || "1".equals(sz3Flag.trim());
            if (!szBench && !sz3Bench) {
                return;
            }
            File root = repoRootFromUserDir();
            File cfg = resolveSzConfig(root);
            if (cfg != null && cfg.isFile()) {
                File sz = resolveSzExecutable(root);
                if (szBench && sz != null) {
                    for (SzFloatPreset preset : SZ_FLOAT_PRESETS) {
                        if (AlgorithmAllowList.shouldSkip(skipAlgorithms, preset.rowName)) {
                            continue;
                        }
                        long[] t = new long[4];
                        double[] loss = new double[2];
                        if (benchSzFloatPreset(columns, sz, cfg, preset, t, loss)) {
                            recordSzRow(preset.rowName, t, totals, csvWriter, skipAlgorithms, loss[0], loss[1]);
                        }
                    }
                }
                File sz3 = resolveSz3Executable(root);
                if (sz3Bench && sz3 != null) {
                    for (SzFloatPreset preset : SZ_FLOAT_PRESETS) {
                        String sz3Name = preset.rowName.replace("sz_", "sz3_");
                        if (AlgorithmAllowList.shouldSkip(skipAlgorithms, sz3Name)) {
                            continue;
                        }
                        long[] t = new long[4];
                        double[] loss = new double[2];
                        if (benchSzFloatPreset(columns, sz3, cfg, preset, t, loss)) {
                            recordSzRow(sz3Name, t, totals, csvWriter, skipAlgorithms, loss[0], loss[1]);
                        }
                    }
                }
            }
        }
    }

    private static void runSzRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        ArrayList<ArrayList<Double>> columns;
        if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
            columns = floatCols;
        } else if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
            columns = BenchmarkColumns.integersAsDoubleColumns(intCols);
        } else {
            columns = new ArrayList<>();
        }
        if (columns.isEmpty()) {
            return;
        }
        final String szFlag = System.getenv("WEB_COMPRESSION_SZ_BENCH");
        final boolean szBench = szFlag == null || szFlag.isEmpty() || "1".equals(szFlag.trim());
        final String sz3Flag = System.getenv("WEB_COMPRESSION_SZ3_BENCH");
        final boolean sz3Bench = sz3Flag == null || sz3Flag.isEmpty() || "1".equals(sz3Flag.trim());
        if (!szBench && !sz3Bench) {
            return;
        }
        File root = repoRootFromUserDir();
        File cfg = resolveSzConfig(root);
        if (cfg == null || !cfg.isFile()) {
            return;
        }
        File sz = resolveSzExecutable(root);
        if (szBench && sz != null) {
            for (SzFloatPreset preset : SZ_FLOAT_PRESETS) {
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, preset.rowName)) {
                    continue;
                }
                long[] t = new long[4];
                double[] loss = new double[2];
                if (benchSzDoublePreset(columns, sz, cfg, preset, t, loss)) {
                    recordSzRow(preset.rowName, t, totals, csvWriter, skipAlgorithms, loss[0], loss[1]);
                }
            }
        }
        File sz3 = resolveSz3Executable(root);
        if (sz3Bench && sz3 != null) {
            for (SzFloatPreset preset : SZ_FLOAT_PRESETS) {
                String sz3Name = preset.rowName.replace("sz_", "sz3_");
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, sz3Name)) {
                    continue;
                }
                long[] t = new long[4];
                double[] loss = new double[2];
                if (benchSzDoublePreset(columns, sz3, cfg, preset, t, loss)) {
                    recordSzRow(sz3Name, t, totals, csvWriter, skipAlgorithms, loss[0], loss[1]);
                }
            }
        }
    }

    private static void recordSzRow(
            String rowName,
            long[] result,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms,
            double lossErr,
            double lossRef)
            throws IOException {
        if (result == null
                || result.length < 4
                || result[0] <= 0
                || AlgorithmAllowList.shouldSkip(skipAlgorithms, rowName)) {
            return;
        }
        if (totals != null) {
            totals.merge(rowName, result, lossErr, lossRef);
        } else if (csvWriter != null) {
            csvWriter.printRecord(rowName, result[0], result[1], result[2], result[3]);
        }
    }

    private static void recordSzRow(
            String rowName,
            long[] result,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms)
            throws IOException {
        recordSzRow(rowName, result, totals, csvWriter, skipAlgorithms, 0.0, 0.0);
    }

    private static Path resolveSzDecompressedPath(Path inputPath, Path compressedPath) {
        Path[] candidates = {
            Path.of(compressedPath.toString() + ".out"),
            Path.of(compressedPath.toString() + "."),
            Path.of(inputPath.toString() + ".out"),
        };
        for (Path p : candidates) {
            try {
                if (Files.isRegularFile(p)) {
                    return p;
                }
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    private static boolean szVerifyDecompressedColumn(
            Path inputPath, Path compressedPath, int n, int valueBytes) {
        Path decPath = resolveSzDecompressedPath(inputPath, compressedPath);
        if (decPath == null) {
            return false;
        }
        try {
            long need = (long) n * (long) valueBytes;
            return Files.size(decPath) >= need;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean szAccumulateFloatColumnLoss(
            List<Float> col, Path inputPath, Path compressedPath, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return szVerifyDecompressedColumn(inputPath, compressedPath, col.size(), 4);
        }
        Path decPath = resolveSzDecompressedPath(inputPath, compressedPath);
        if (decPath == null) {
            return false;
        }
        try {
            byte[] decBytes = Files.readAllBytes(decPath);
            int n = col.size();
            List<Double> origVals = new ArrayList<>(n);
            List<Double> decVals = new ArrayList<>(n);
            for (float f : col) {
                origVals.add((double) f);
            }
            ByteBuffer db = ByteBuffer.wrap(decBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n && db.remaining() >= 4; i++) {
                decVals.add((double) db.getFloat());
            }
            if (decVals.size() != n) {
                return false;
            }
            BenchmarkLoss.accumulateDoubleListLoss(origVals, decVals, lossAcc);
            return true;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean szAccumulateDoubleColumnLoss(
            List<Double> col, Path inputPath, Path compressedPath, double[] lossAcc) {
        if (lossAcc == null || lossAcc.length < 2) {
            return szVerifyDecompressedColumn(inputPath, compressedPath, col.size(), 8);
        }
        Path decPath = resolveSzDecompressedPath(inputPath, compressedPath);
        if (decPath == null) {
            return false;
        }
        try {
            byte[] decBytes = Files.readAllBytes(decPath);
            int n = col.size();
            List<Double> decVals = new ArrayList<>(n);
            ByteBuffer db = ByteBuffer.wrap(decBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n && db.remaining() >= 8; i++) {
                decVals.add(db.getDouble());
            }
            if (decVals.size() != n) {
                return false;
            }
            BenchmarkLoss.accumulateDoubleListLoss(col, decVals, lossAcc);
            return true;
        } catch (Exception ignored) {
            return false;
        }
    }

    static File repoRootFromUserDir() {
        String repo = System.getenv("WEB_COMPRESSION_REPO_ROOT");
        if (repo != null && !repo.isEmpty()) {
            return new File(repo).getAbsoluteFile();
        }
        File cwd = new File(System.getProperty("user.dir")).getAbsoluteFile();
        if ("backend".equalsIgnoreCase(cwd.getName())) {
            return cwd.getParentFile();
        }
        return cwd;
    }

    static File resolveSzExecutable(File root) {
        File env = NativeCliExecutable.fromEnv("WEB_COMPRESSION_SZ");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root,
                "external/compression/SZ/example/sz",
                "external/compression/SZ/build/bin/sz");
    }

    static File resolveSzConfig(File root) {
        String cfgEnv = System.getenv("WEB_COMPRESSION_SZ_CONFIG");
        if (cfgEnv != null && !cfgEnv.isEmpty()) {
            File f = new File(cfgEnv);
            if (f.isFile()) {
                return f;
            }
        }
        String sz3Cfg = System.getenv("WEB_COMPRESSION_SZ3_CONFIG");
        if (sz3Cfg != null && !sz3Cfg.isEmpty()) {
            File f = new File(sz3Cfg);
            if (f.isFile()) {
                return f;
            }
        }
        File legacy = new File(root, "external/compression/SZ/example/sz.config");
        if (legacy.isFile()) {
            return legacy;
        }
        return new File(root, "external/compression/SZ3/sz.config");
    }

    static File resolveSz3Executable(File root) {
        File env = NativeCliExecutable.fromEnv("WEB_COMPRESSION_SZ3");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root,
                "external/compression/SZ3/build/tools/sz3/sz3",
                "external/compression/SZ3/build/tools/sz3/Release/sz3");
    }

    /**
     * Aggregate one SZ preset over all float columns (same pattern as other benchmark rows).
     */
    static boolean benchSzFloatPreset(
            List<ArrayList<Float>> columns,
            File sz,
            File cfg,
            SzFloatPreset preset,
            long[] out) {
        return benchSzFloatPreset(columns, sz, cfg, preset, out, null);
    }

    static boolean benchSzFloatPreset(
            List<ArrayList<Float>> columns,
            File sz,
            File cfg,
            SzFloatPreset preset,
            long[] out,
            double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_sz_float");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        final File repoRoot = repoRootFromUserDir();
        try {
            for (List<Float> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 4];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : col) {
                    bb.putFloat(f);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".dat");
                Path outSz = Path.of(in + ".sz");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(outSz);

                    List<String> encCmd = new ArrayList<>();
                    encCmd.add(sz.getAbsolutePath());
                    encCmd.add("-z");
                    encCmd.add("-f");
                    encCmd.add("-c");
                    encCmd.add(cfg.getAbsolutePath());
                    encCmd.add("-M");
                    encCmd.add(preset.errMode);
                    encCmd.add(preset.boundFlag);
                    encCmd.add(preset.boundValue);
                    encCmd.add("-i");
                    encCmd.add(in.toAbsolutePath().toString());
                    encCmd.add("-1");
                    encCmd.add(Integer.toString(n));

                    long t0 = System.nanoTime();
                    ProcessBuilder enc = new ProcessBuilder(encCmd);
                    NativeCliExecutable.configureNativeCliProcess(enc, sz, repoRoot);
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    try {
                        if (!p1.waitFor(120, TimeUnit.SECONDS) || p1.exitValue() != 0) {
                            p1.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(outSz)) {
                        continue;
                    }
                    long csize = Files.size(outSz);

                    long t2 = System.nanoTime();
                    ProcessBuilder dec = new ProcessBuilder(
                            sz.getAbsolutePath(),
                            "-x",
                            "-f",
                            "-s",
                            outSz.toAbsolutePath().toString(),
                            "-1",
                            Integer.toString(n));
                    NativeCliExecutable.configureNativeCliProcess(dec, sz, repoRoot);
                    dec.redirectError(ProcessBuilder.Redirect.DISCARD);
                    dec.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = dec.start();
                    try {
                        if (!p2.waitFor(120, TimeUnit.SECONDS) || p2.exitValue() != 0) {
                            p2.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    if (!szAccumulateFloatColumnLoss(col, in, outSz, lossAcc)) {
                        continue;
                    }

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Path decPath = resolveSzDecompressedPath(in, outSz);
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(outSz);
                        if (decPath != null) {
                            Files.deleteIfExists(decPath);
                        }
                    } catch (Exception ignored) {
                    }
                }
            }
        } catch (Exception ignored) {
            out[0] = 0;
            return false;
        } finally {
            try {
                deleteRecursive(tmpDir);
            } catch (Exception ignored) {
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return true;
    }

    static boolean benchSzDoublePreset(
            List<ArrayList<Double>> columns,
            File sz,
            File cfg,
            SzFloatPreset preset,
            long[] out) {
        return benchSzDoublePreset(columns, sz, cfg, preset, out, null);
    }

    static boolean benchSzDoublePreset(
            List<ArrayList<Double>> columns,
            File sz,
            File cfg,
            SzFloatPreset preset,
            long[] out,
            double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_sz_double");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        final File repoRoot = repoRootFromUserDir();
        try {
            for (List<Double> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 8];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (Double d : col) {
                    bb.putDouble(d == null ? Double.NaN : d);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".dat");
                Path outSz = Path.of(in + ".sz");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(outSz);

                    List<String> encCmd = new ArrayList<>();
                    encCmd.add(sz.getAbsolutePath());
                    encCmd.add("-z");
                    encCmd.add("-d");
                    encCmd.add("-c");
                    encCmd.add(cfg.getAbsolutePath());
                    encCmd.add("-M");
                    encCmd.add(preset.errMode);
                    encCmd.add(preset.boundFlag);
                    encCmd.add(preset.boundValue);
                    encCmd.add("-i");
                    encCmd.add(in.toAbsolutePath().toString());
                    encCmd.add("-1");
                    encCmd.add(Integer.toString(n));

                    long t0 = System.nanoTime();
                    ProcessBuilder enc = new ProcessBuilder(encCmd);
                    NativeCliExecutable.configureNativeCliProcess(enc, sz, repoRoot);
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    try {
                        if (!p1.waitFor(120, TimeUnit.SECONDS) || p1.exitValue() != 0) {
                            p1.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(outSz)) {
                        continue;
                    }
                    long csize = Files.size(outSz);

                    long t2 = System.nanoTime();
                    ProcessBuilder dec = new ProcessBuilder(
                            sz.getAbsolutePath(),
                            "-x",
                            "-d",
                            "-s",
                            outSz.toAbsolutePath().toString(),
                            "-1",
                            Integer.toString(n));
                    NativeCliExecutable.configureNativeCliProcess(dec, sz, repoRoot);
                    dec.redirectError(ProcessBuilder.Redirect.DISCARD);
                    dec.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = dec.start();
                    try {
                        if (!p2.waitFor(120, TimeUnit.SECONDS) || p2.exitValue() != 0) {
                            p2.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    if (!szAccumulateDoubleColumnLoss(col, in, outSz, lossAcc)) {
                        continue;
                    }

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Path decPath = resolveSzDecompressedPath(in, outSz);
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(outSz);
                        if (decPath != null) {
                            Files.deleteIfExists(decPath);
                        }
                    } catch (Exception ignored) {
                    }
                }
            }
        } catch (Exception ignored) {
            out[0] = 0;
            return false;
        } finally {
            try {
                deleteRecursive(tmpDir);
            } catch (Exception ignored) {
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return true;
    }

    private static void deleteRecursive(Path root) throws java.io.IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (var walk = Files.walk(root)) {
            walk.sorted((a, b) -> -a.compareTo(b)).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (java.io.IOException ignored) {
                }
            });
        }
    }
}
