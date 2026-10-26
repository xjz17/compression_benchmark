package org.example;

import org.apache.commons.csv.CSVPrinter;
import org.apache.tsfile.enums.TSDataType;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.example.buff.BuffKernelNative;
import org.example.buff.BuffSplitDoubleColumnCodec;

/**
 * BUFF (VLDB 2021) via the upstream Rust {@code comp_profiler} binary under
 * {@code external/compression/buff/database} — see
 * <a href="https://github.com/Tranway1/buff">Tranway1/buff</a>.
 *
 * <p><b>In-process Java core (bounded double + split packing):</b> see
 * {@link org.example.buff.BuffPrecisionBound} and {@link org.example.buff.BuffSplitDoubleColumnCodec},
 * ported from upstream {@code prec_double.rs} / {@code split_double.rs}; the Rust subprocess rows require
 * {@code comp_profiler}; {@code buff_float_kernel_native} is in-process only.
 *
 * <p>Build (toolchain nightly, from {@code database/}): {@code cargo +nightly build --release --bin comp_profiler}.
 * Opt in Rust rows with {@code WEB_COMPRESSION_BUFF_BENCH=1}; override binary with {@code WEB_COMPRESSION_COMP_PROFILER}.
 *
 * <p>{@code buff_float_simd}: on x86_64 the upstream {@code buff-simd} (AVX2) path is used; on AArch64 (e.g. M1)
 * the same report key uses the portable {@code buff} pipeline in {@code comp_profiler}, because the upstream
 * crate’s SIMD kernels are x86-only. Override with {@code WEB_COMPRESSION_BUFF_SIMD_MODE=buff-simd} or {@code buff}.
 *
 * <p>{@code buff_float_kernel_native}: set {@code WEB_COMPRESSION_BUFF_KERNEL_NATIVE=1} — no {@code comp_profiler}
 * needed (uses {@link BuffSplitDoubleColumnCodec} + {@link BuffKernelNative}). */
public final class ResearchBuffCliBench {

    private ResearchBuffCliBench() {
    }

    private static final String ROW_SCALAR = "buff_scalar";
    private static final String ROW_SIMD = "buff_simd";
    private static final String ROW_KERNEL_NATIVE = "buff_kernel_native";
    private static final int DEFAULT_INT_SCALE = 10000;
    private static final double DEFAULT_PRED = 1.1509;

    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runBuffRows(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows(
            TSDataType tsType,
            @SuppressWarnings("unused") ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runBuffRows(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    public static void accumulateRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runBuffRows64(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runBuffRows64(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    private static void runBuffRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        ArrayList<ArrayList<Float>> columns =
                BenchmarkColumns.floatColumnsForCodecBench(tsType, intCols, floatCols);
        if (columns.isEmpty()) {
            return;
        }
        boolean buffBench = "1".equals(System.getenv("WEB_COMPRESSION_BUFF_BENCH"));
        boolean kernelNative = "1".equals(System.getenv("WEB_COMPRESSION_BUFF_KERNEL_NATIVE"));
        if (!buffBench && !kernelNative) {
            return;
        }
        int scale = DEFAULT_INT_SCALE;
        String es = System.getenv("WEB_COMPRESSION_BUFF_INT_SCALE");
        if (es != null && !es.isEmpty()) {
            try {
                scale = Integer.parseInt(es.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        double pred = DEFAULT_PRED;
        String ep = System.getenv("WEB_COMPRESSION_BUFF_PRED");
        if (ep != null && !ep.isEmpty()) {
            try {
                pred = Double.parseDouble(ep.trim());
            } catch (NumberFormatException ignored) {
            }
        }

        if (buffBench) {
            File root = ResearchNativeCliBench.repoRootFromUserDir();
            File profiler = resolveCompProfiler(root);
            File databaseDir = new File(root, "external/compression/buff/database");
            if (profiler != null && databaseDir.isDirectory()) {
                long[] t = new long[4];
                recordBuffRow(
                        ROW_SCALAR,
                        t,
                        skipAlgorithms,
                        totals,
                        csvWriter,
                        !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SCALAR)
                                && benchBuff(columns, profiler, databaseDir, scale, pred, "buff", t));
                String simdMode = resolveBuffSimdProfilerMode();
                long[] ts = new long[4];
                if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SIMD) && "buff".equals(simdMode) && t[0] > 0) {
                    recordBuffRow(ROW_SIMD, t, skipAlgorithms, totals, csvWriter, true);
                } else {
                    recordBuffRow(
                            ROW_SIMD,
                            ts,
                            skipAlgorithms,
                            totals,
                            csvWriter,
                            !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SIMD)
                                    && benchBuff(columns, profiler, databaseDir, scale, pred, simdMode, ts));
                }
            }
        }
        if (kernelNative && !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_KERNEL_NATIVE)) {
            long[] kn = new long[4];
            recordBuffRow(
                    ROW_KERNEL_NATIVE,
                    kn,
                    skipAlgorithms,
                    totals,
                    csvWriter,
                    benchBuffKernelNative(columns, scale, pred, kn));
        }
    }

    private static void runBuffRows64(
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
        boolean buffBench = "1".equals(System.getenv("WEB_COMPRESSION_BUFF_BENCH"));
        boolean kernelNative = "1".equals(System.getenv("WEB_COMPRESSION_BUFF_KERNEL_NATIVE"));
        if (!buffBench && !kernelNative) {
            return;
        }
        int scale = DEFAULT_INT_SCALE;
        String es = System.getenv("WEB_COMPRESSION_BUFF_INT_SCALE");
        if (es != null && !es.isEmpty()) {
            try {
                scale = Integer.parseInt(es.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        double pred = DEFAULT_PRED;
        String ep = System.getenv("WEB_COMPRESSION_BUFF_PRED");
        if (ep != null && !ep.isEmpty()) {
            try {
                pred = Double.parseDouble(ep.trim());
            } catch (NumberFormatException ignored) {
            }
        }

        if (buffBench) {
            File root = ResearchNativeCliBench.repoRootFromUserDir();
            File profiler = resolveCompProfiler(root);
            File databaseDir = new File(root, "external/compression/buff/database");
            if (profiler != null && databaseDir.isDirectory()) {
                long[] t = new long[4];
                recordBuffRow(
                        ROW_SCALAR,
                        t,
                        skipAlgorithms,
                        totals,
                        csvWriter,
                        !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SCALAR)
                                && benchBuffDouble(columns, profiler, databaseDir, scale, pred, "buff", t));
                String simdMode = resolveBuffSimdProfilerMode();
                long[] ts = new long[4];
                if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SIMD) && "buff".equals(simdMode) && t[0] > 0) {
                    recordBuffRow(ROW_SIMD, t, skipAlgorithms, totals, csvWriter, true);
                } else {
                    recordBuffRow(
                            ROW_SIMD,
                            ts,
                            skipAlgorithms,
                            totals,
                            csvWriter,
                            !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_SIMD)
                                    && benchBuffDouble(columns, profiler, databaseDir, scale, pred, simdMode, ts));
                }
            }
        }
        if (kernelNative && !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_KERNEL_NATIVE)) {
            long[] kn = new long[4];
            recordBuffRow(
                    ROW_KERNEL_NATIVE,
                    kn,
                    skipAlgorithms,
                    totals,
                    csvWriter,
                    benchBuffKernelNativeDouble(columns, scale, pred, kn));
        }
    }

    private static void recordBuffRow(
            String rowName,
            long[] result,
            Set<String> skipAlgorithms,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            boolean ok)
            throws IOException {
        if (!ok || result == null || result.length < 4 || result[0] <= 0) {
            return;
        }
        if (totals != null) {
            totals.merge(rowName, result);
        } else if (csvWriter != null) {
            csvWriter.printRecord(rowName, result[0], result[1], result[2], result[3]);
        }
    }

    static boolean benchBuffKernelNative(
            List<ArrayList<Float>> columns, int intScale, double pred, long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        for (List<Float> col : columns) {
            int n = col.size();
            if (n < 1) {
                continue;
            }
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                d[i] = col.get(i);
            }
            orig += BenchmarkWireFormat.columnOriginalSize(n, true);
            long t0 = System.nanoTime();
            byte[] packed = BuffSplitDoubleColumnCodec.encodeDoubles(d, intScale);
            long t1 = System.nanoTime();
            BuffSplitDoubleColumnCodec.DecodedColumn decoded = BuffSplitDoubleColumnCodec.decodeDoubles(packed);
            long t2 = System.nanoTime();
            if (!rawDoubleArraysEqual(d, decoded.values)) continue;
            comp += packed.length;
            ct += (t1 - t0);
            dt += (t2 - t1);
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    static boolean benchBuffKernelNativeDouble(
            List<? extends List<Double>> columns, int intScale, double pred, long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        for (List<? extends Double> col : columns) {
            int n = col.size();
            if (n < 1) {
                continue;
            }
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                Double v = col.get(i);
                d[i] = v == null ? Double.NaN : v;
            }
            orig += BenchmarkWireFormat.columnOriginalSize(n, true);
            long t0 = System.nanoTime();
            byte[] packed = BuffSplitDoubleColumnCodec.encodeDoubles(d, intScale);
            long t1 = System.nanoTime();
            BuffSplitDoubleColumnCodec.DecodedColumn decoded = BuffSplitDoubleColumnCodec.decodeDoubles(packed);
            long t2 = System.nanoTime();
            if (!rawDoubleArraysEqual(d, decoded.values)) continue;
            comp += packed.length;
            ct += (t1 - t0);
            dt += (t2 - t1);
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    private static boolean rawDoubleArraysEqual(double[] left, double[] right) {
        if (left.length != right.length) return false;
        for (int i = 0; i < left.length; i++) {
            if (Double.doubleToRawLongBits(left[i]) != Double.doubleToRawLongBits(right[i])) return false;
        }
        return true;
    }

    /** {@code buff-simd} on x86; {@code buff} on AArch64 unless env overrides. */
    static String resolveBuffSimdProfilerMode() {
        String o = System.getenv("WEB_COMPRESSION_BUFF_SIMD_MODE");
        if (o != null && !o.isEmpty()) {
            String m = o.trim().toLowerCase(Locale.ROOT);
            if ("buff-simd".equals(m) || "buff".equals(m)) {
                return m;
            }
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        if ("x86_64".equals(arch) || "amd64".equals(arch)) {
            return "buff-simd";
        }
        if ("aarch64".equals(arch) || "arm64".equals(arch)) {
            return "buff";
        }
        return "buff";
    }

    static File resolveCompProfiler(File root) {
        String env = System.getenv("WEB_COMPRESSION_COMP_PROFILER");
        if (env != null && !env.isEmpty()) {
            File f = new File(env);
            if (f.isFile() && f.canExecute()) {
                return f;
            }
        }
        for (String rel :
                new String[] {
                    "external/compression/buff/database/target/release/comp_profiler",
                    "external/compression/buff/database/target/debug/comp_profiler",
                }) {
            File cand = new File(root, rel);
            if (cand.isFile() && cand.canExecute()) {
                return cand;
            }
        }
        return null;
    }

    static boolean benchBuff(
            List<ArrayList<Float>> columns,
            File profiler,
            File databaseCwd,
            int intScale,
            double pred,
            String profilerMode,
            long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_buff_float");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        try {
            String mode = profilerMode != null && !profilerMode.isEmpty() ? profilerMode : "buff";
            for (List<Float> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                long orgCol = (long) n * 8L;
                Path csvPath = tmpDir.resolve("in-" + System.nanoTime() + ".csv");
                try {
                    try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
                        for (float f : col) {
                            w.write(Double.toString(f));
                            w.newLine();
                        }
                    }
                    ProcessBuilder pb =
                            new ProcessBuilder(
                                    profiler.getAbsolutePath(),
                                    csvPath.toAbsolutePath().toString(),
                                    mode,
                                    Integer.toString(intScale),
                                    Double.toString(pred));
                    pb.directory(databaseCwd);
                    pb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    Process proc = pb.start();
                    String outText;
                    try (Scanner sc = new Scanner(proc.getInputStream())) {
                        sc.useDelimiter("\\A");
                        outText = sc.hasNext() ? sc.next() : "";
                    }
                    if (!proc.waitFor(600, TimeUnit.SECONDS) || proc.exitValue() != 0) {
                        proc.destroyForcibly();
                        continue;
                    }
                    PerformanceParsed p = parsePerformanceFields(outText);
                    if (p == null || p.ratio <= 0 || p.encMbps <= 0 || p.decMbps <= 0) {
                        continue;
                    }
                    long compCol = (long) (p.ratio * orgCol);
                    long encNs = (long) ((orgCol * 1_000_000_000.0) / (p.encMbps * 1024.0 * 1024.0));
                    long decNs = (long) ((orgCol * 1_000_000_000.0) / (p.decMbps * 1024.0 * 1024.0));
                    if (encNs <= 0 || decNs <= 0 || compCol <= 0) {
                        continue;
                    }

                    orig += orgCol;
                    comp += compCol;
                    ct += encNs;
                    dt += decNs;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(csvPath);
                    } catch (IOException ignored) {
                    }
                }
            }
        } finally {
            deleteRecursiveQuiet(tmpDir);
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    static boolean benchBuffDouble(
            List<? extends List<Double>> columns,
            File profiler,
            File databaseCwd,
            int intScale,
            double pred,
            String profilerMode,
            long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_buff_double");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        try {
            String mode = profilerMode != null && !profilerMode.isEmpty() ? profilerMode : "buff";
            for (List<? extends Double> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                long orgCol = (long) n * 8L;
                Path csvPath = tmpDir.resolve("in-" + System.nanoTime() + ".csv");
                try {
                    try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
                        for (Double d : col) {
                            w.write(Double.toString(d == null ? Double.NaN : d));
                            w.newLine();
                        }
                    }
                    ProcessBuilder pb =
                            new ProcessBuilder(
                                    profiler.getAbsolutePath(),
                                    csvPath.toAbsolutePath().toString(),
                                    mode,
                                    Integer.toString(intScale),
                                    Double.toString(pred));
                    pb.directory(databaseCwd);
                    pb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    Process proc = pb.start();
                    String outText;
                    try (Scanner sc = new Scanner(proc.getInputStream())) {
                        sc.useDelimiter("\\A");
                        outText = sc.hasNext() ? sc.next() : "";
                    }
                    if (!proc.waitFor(600, TimeUnit.SECONDS) || proc.exitValue() != 0) {
                        proc.destroyForcibly();
                        continue;
                    }
                    PerformanceParsed p = parsePerformanceFields(outText);
                    if (p == null || p.ratio <= 0 || p.encMbps <= 0 || p.decMbps <= 0) {
                        continue;
                    }
                    long compCol = (long) (p.ratio * orgCol);
                    long encNs = (long) ((orgCol * 1_000_000_000.0) / (p.encMbps * 1024.0 * 1024.0));
                    long decNs = (long) ((orgCol * 1_000_000_000.0) / (p.decMbps * 1024.0 * 1024.0));
                    if (encNs <= 0 || decNs <= 0 || compCol <= 0) {
                        continue;
                    }
                    orig += orgCol;
                    comp += compCol;
                    ct += encNs;
                    dt += decNs;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(csvPath);
                    } catch (IOException ignored) {
                    }
                }
            }
        } finally {
            deleteRecursiveQuiet(tmpDir);
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    private static final class PerformanceParsed {
        final double ratio;
        final double encMbps;
        final double decMbps;

        PerformanceParsed(double ratio, double encMbps, double decMbps) {
            this.ratio = ratio;
            this.encMbps = encMbps;
            this.decMbps = decMbps;
        }
    }

    private static PerformanceParsed parsePerformanceFields(String stdout) {
        if (stdout == null) {
            return null;
        }
        String line = null;
        for (String ln : stdout.split("\n")) {
            if (ln.startsWith("Performance:")) {
                line = ln;
                break;
            }
        }
        if (line == null) {
            return null;
        }
        String rest = line.substring("Performance:".length());
        String[] parts = rest.split(",", -1);
        if (parts.length < 10) {
            return null;
        }
        int n = parts.length;
        try {
            double ratio = Double.parseDouble(parts[n - 7].trim());
            double enc = Double.parseDouble(parts[n - 6].trim());
            double dec = Double.parseDouble(parts[n - 5].trim());
            return new PerformanceParsed(ratio, enc, dec);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static void deleteRecursiveQuiet(Path root) {
        try {
            if (!Files.exists(root)) {
                return;
            }
            try (var walk = Files.walk(root)) {
                walk.sorted((a, b) -> -a.compareTo(b))
                        .forEach(
                                p -> {
                                    try {
                                        Files.deleteIfExists(p);
                                    } catch (IOException ignored) {
                                    }
                                });
            }
        } catch (IOException ignored) {
        }
    }
}
