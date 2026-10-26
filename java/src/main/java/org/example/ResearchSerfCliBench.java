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
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Serf (lossy XOR) via the upstream {@code sfz} CLI when built under
 * {@code external/compression/Serf/build/serf_cmd/sfz}, or {@code COMPRESSION_BENCHMARK_SFZ}.
 * Per-call timeout follows {@code COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC} (default 300s; {@code 0}
 * = unlimited), aligned with {@code cpp/src/research_cli.cpp}.
 */
public final class ResearchSerfCliBench {

    private ResearchSerfCliBench() {
    }

    private static final class SerfAbsPreset {
        final String rowName;
        final String bound;

        SerfAbsPreset(String rowName, String bound) {
            this.rowName = rowName;
            this.bound = bound;
        }
    }

    private static final SerfAbsPreset[] SERF_PRESETS = {
            new SerfAbsPreset("serf_ABS_1e-1", "1e-1"),
            new SerfAbsPreset("serf_ABS_1e-2", "1e-2"),
            new SerfAbsPreset("serf_ABS_1e-3", "1e-3"),
            new SerfAbsPreset("serf_ABS_1e-4", "1e-4"),
    };

    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runSerfRows(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows(
            TSDataType tsType,
            @SuppressWarnings("unused") ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runSerfRows(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    private static void runSerfRows(
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
        // Subprocess + text I/O per column × preset: extremely slow for staticdata; opt in with
        // COMPRESSION_BENCHMARK_SERF_BENCH=1 or leave unset when using upload-only Serf timings.
        if (!"1".equals(System.getenv("COMPRESSION_BENCHMARK_SERF_BENCH"))) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File sfz = resolveSfzExecutable(root);
        if (sfz == null) {
            return;
        }
        for (SerfAbsPreset p : SERF_PRESETS) {
            if (AlgorithmAllowList.shouldSkip(skipAlgorithms, p.rowName)) {
                continue;
            }
            long[] t = new long[4];
            double[] loss = new double[2];
            if (benchSerfAbs(columns, sfz, p.bound, t, loss) && t[0] > 0) {
                if (totals != null) {
                    totals.merge(p.rowName, t, loss[0], loss[1]);
                } else if (csvWriter != null) {
                    csvWriter.printRecord(p.rowName, t[0], t[1], t[2], t[3]);
                }
            }
        }
    }

    /**
     * Per compress/decompress call; matches {@code cpp/src/research_cli.cpp} ({@code
     * COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC}). {@code 0} = unlimited wait.
     */
    static long serfShellTimeoutSeconds() {
        String e = System.getenv("COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC");
        if (e == null || e.isEmpty()) {
            return 300L;
        }
        try {
            long v = Long.parseLong(e.trim());
            if (v <= 0) {
                return 0L;
            }
            return Math.min(v, 86400L);
        } catch (NumberFormatException ex) {
            return 300L;
        }
    }

    static File resolveSfzExecutable(File root) {
        File env = NativeCliExecutable.fromEnv("COMPRESSION_BENCHMARK_SFZ");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root, "external/compression/Serf/build/serf_cmd/sfz");
    }

    static boolean benchSerfAbs(
            List<ArrayList<Float>> columns, File sfz, String boundStr, long[] out) {
        return benchSerfAbs(columns, sfz, boundStr, out, null);
    }

    static boolean benchSerfAbs(
            List<ArrayList<Float>> columns, File sfz, String boundStr, long[] out, double[] lossAcc) {
        double bound = Double.parseDouble(boundStr);
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_serf_float");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        try {
            for (List<Float> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                Path inTxt = tmpDir.resolve("in-" + System.nanoTime() + ".txt");
                Path outBin = tmpDir.resolve("cmp-" + System.nanoTime() + ".bin");
                Path outTxt = tmpDir.resolve("dec-" + System.nanoTime() + ".txt");
                try {
                    try (BufferedWriter w = Files.newBufferedWriter(inTxt)) {
                        for (float f : col) {
                            w.write(Float.toString(f));
                            w.newLine();
                        }
                    }
                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    sfz.getAbsolutePath(),
                                    "compress",
                                    boundStr,
                                    inTxt.toAbsolutePath().toString(),
                                    outBin.toAbsolutePath().toString());
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    long sec = serfShellTimeoutSeconds();
                    boolean encOk =
                            sec == 0
                                    ? p1.waitFor() == 0
                                    : p1.waitFor(sec, TimeUnit.SECONDS) && p1.exitValue() == 0;
                    if (!encOk) {
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(outBin)) {
                        continue;
                    }
                    long csize = Files.size(outBin);

                    long t2 = System.nanoTime();
                    ProcessBuilder dec =
                            new ProcessBuilder(
                                    sfz.getAbsolutePath(),
                                    "decompress",
                                    boundStr,
                                    outBin.toAbsolutePath().toString(),
                                    outTxt.toAbsolutePath().toString());
                    dec.redirectError(ProcessBuilder.Redirect.DISCARD);
                    dec.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = dec.start();
                    boolean decOk =
                            sec == 0
                                    ? p2.waitFor() == 0
                                    : p2.waitFor(sec, TimeUnit.SECONDS) && p2.exitValue() == 0;
                    if (!decOk) {
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    try (Scanner sc = new Scanner(outTxt)) {
                        int i = 0;
                        while (sc.hasNextLine() && i < n) {
                            double dv = Double.parseDouble(sc.nextLine().trim());
                            float got = (float) dv;
                            float exp = col.get(i);
                            if (lossAcc != null) {
                                lossAcc[0] += Math.abs((double) exp - (double) got);
                                lossAcc[1] += Math.abs((double) exp);
                            }
                            if (Math.abs(got - exp) > bound * 1.0001 + 1e-6) {
                                i = -1;
                                break;
                            }
                            i++;
                        }
                        if (i != n) {
                            continue;
                        }
                    }

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(inTxt);
                        Files.deleteIfExists(outBin);
                        Files.deleteIfExists(outTxt);
                    } catch (Exception ignored) {
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
