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
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Disabled legacy FPCompress CLI helper. Runtime {@code fpcompress_dpratio} and
 * {@code fpcompress_dpspeed} rows are emitted by {@link NativeLzFamilyBench}.
 */
public final class ResearchFpcompressCliBench {

    private enum Variant {
        DPRATIO("fpcompress_dpratio", true),
        DPSPEED("fpcompress_dpspeed", false);

        final String row;
        final boolean ratio;

        Variant(String row, boolean ratio) {
            this.row = row;
            this.ratio = ratio;
        }
    }

    private static final String SRC_TREE = "double_src";

    private ResearchFpcompressCliBench() {}

    public static void accumulateRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        return;
    }

    public static void printRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            CSVPrinter csvWriter)
            throws IOException {
        return;
    }

    private static void runAll64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms)
            throws IOException {
        if (!benchEnabled()) {
            return;
        }
        File repoRoot = ResearchNativeCliBench.repoRootFromUserDir();
        ArrayList<ArrayList<Double>> dpCols = null;
        if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
            dpCols = BenchmarkColumns.integersAsDoubleColumns(intCols);
        } else if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
            dpCols = floatCols;
        }
        if (dpCols == null || dpCols.isEmpty()) {
            return;
        }
        for (Variant v : Variant.values()) {
            if (AlgorithmAllowList.shouldSkip(skipAlgorithms, v.row)) {
                continue;
            }
            long[] t = new long[4];
            double[] loss = new double[2];
            boolean ok = benchDpDoubleColumns(dpCols, repoRoot, v, t, loss);
            emitRow(v.row, ok, t, loss, totals, csvWriter);
        }
    }

    private static void emitRow(
            String row,
            boolean ok,
            long[] t,
            double[] loss,
            BenchmarkTotals totals,
            CSVPrinter csvWriter)
            throws IOException {
        if (!ok || t[0] <= 0) {
            return;
        }
        if (totals != null) {
            totals.merge(row, t, loss[0], loss[1]);
        } else if (csvWriter != null) {
            csvWriter.printRecord(row, t[0], t[1], t[2], t[3]);
        }
    }

    static boolean benchEnabled() {
        final String flag = System.getenv("COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH");
        if (flag != null && !flag.isEmpty() && !"1".equals(flag.trim())) {
            return false;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        return resolveExe(root, SRC_TREE, "ratio-cpu-compress") != null
                && resolveExe(root, SRC_TREE, "ratio-cpu-decompress") != null;
    }

    static File resolveBinRoot(File repoRoot) {
        String env = System.getenv("COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT");
        if (env != null && !env.isBlank()) {
            return new File(env.trim());
        }
        return new File(repoRoot, "external/runtime/fpcompress");
    }

    static File resolveExe(File repoRoot, String srcTree, String stem) {
        File binRoot = resolveBinRoot(repoRoot);
        File hit = NativeCliExecutable.firstRunnable(binRoot, srcTree + "/bin/" + stem);
        if (hit != null) {
            return hit;
        }
        hit = NativeCliExecutable.firstRunnable(binRoot, srcTree + "/" + stem);
        if (hit != null) {
            return hit;
        }
        File upstream = new File(repoRoot, "external/compression/FPCompress");
        hit = NativeCliExecutable.firstRunnable(upstream, srcTree + "/bin/" + stem);
        if (hit != null) {
            return hit;
        }
        return NativeCliExecutable.firstRunnable(upstream, srcTree + "/" + stem);
    }

    private static boolean benchDpDoubleColumns(
            List<ArrayList<Double>> columns, File repoRoot, Variant v, long[] out, double[] lossAcc) {
        File comp = resolveExe(repoRoot, SRC_TREE, v.ratio ? "ratio-cpu-compress" : "speed-cpu-compress");
        File decomp =
                resolveExe(repoRoot, SRC_TREE, v.ratio ? "ratio-cpu-decompress" : "speed-cpu-decompress");
        if (comp == null || decomp == null) {
            out[0] = 0;
            return false;
        }
        long orig = 0;
        long ctot = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_fpcompress");
        } catch (IOException e) {
            out[0] = 0;
            return false;
        }
        try {
            for (List<Double> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 8];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (double dv : col) {
                    bb.putDouble(dv);
                }
                long[] one = new long[4];
                if (!benchOneColumn(raw, n, comp, decomp, repoRoot, tmpDir, lossAcc, one)) {
                    continue;
                }
                orig += one[0];
                ctot += one[1];
                ct += one[2];
                dt += one[3];
            }
        } finally {
            deleteRecursiveQuiet(tmpDir);
        }
        out[0] = orig;
        out[1] = ctot;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    private static boolean benchOneColumn(
            byte[] raw,
            int n,
            File comp,
            File decomp,
            File repoRoot,
            Path tmpDir,
            double[] lossAcc,
            long[] oneOut) {
        try {
            Path in = tmpDir.resolve("in-" + System.nanoTime() + ".bin");
            Path compPath = tmpDir.resolve("comp-" + System.nanoTime() + ".bin");
            Path out = tmpDir.resolve("out-" + System.nanoTime() + ".bin");
            Files.write(in, raw);
            long t0 = System.nanoTime();
            if (!runCli(comp, in, compPath, repoRoot)) {
                return false;
            }
            long t1 = System.nanoTime();
            if (!Files.isRegularFile(compPath)) {
                return false;
            }
            long csize = Files.size(compPath);
            long t2 = System.nanoTime();
            if (!runCli(decomp, compPath, out, repoRoot)) {
                return false;
            }
            long t3 = System.nanoTime();
            if (!Files.isRegularFile(out) || Files.size(out) != raw.length) {
                return false;
            }
            byte[] back = Files.readAllBytes(out);
            if (!Arrays.equals(raw, back)) {
                return false;
            }
            BenchmarkLoss.accumulateByteLoss(raw, back, lossAcc);
            oneOut[0] = BenchmarkWireFormat.columnOriginalSize(n, true);
            oneOut[1] = csize;
            oneOut[2] = t1 - t0;
            oneOut[3] = t3 - t2;
            Files.deleteIfExists(in);
            Files.deleteIfExists(compPath);
            Files.deleteIfExists(out);
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    private static boolean runCli(File exe, Path input, Path output, File repoRoot) {
        try {
            ProcessBuilder pb =
                    new ProcessBuilder(
                            exe.getAbsolutePath(),
                            input.toAbsolutePath().toString(),
                            output.toAbsolutePath().toString());
            pb.redirectError(ProcessBuilder.Redirect.DISCARD);
            pb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
            NativeCliExecutable.configureNativeCliProcess(pb, exe, repoRoot);
            Process p = pb.start();
            if (!p.waitFor(300, TimeUnit.SECONDS) || p.exitValue() != 0) {
                p.destroyForcibly();
                return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static void deleteRecursiveQuiet(Path root) {
        if (root == null || !Files.exists(root)) {
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
        } catch (IOException ignored) {
        }
    }
}
