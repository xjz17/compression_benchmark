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
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * ZFP on float64 column wire via the LLNL {@code zfp} CLI ({@code -d}).
 * Int benchmarks cast int64 columns to double ({@link BenchmarkColumns#doubleColumnsForCodecBench}).
 *
 * <p>Rows: {@code ext_zfp_zfpy} (reversible {@code -R}) and {@code ext_zfp_tol_*} ({@code -a}).
 * Override binary with {@code WEB_COMPRESSION_ZFP}; disable with {@code WEB_COMPRESSION_ZFP_BENCH=0}.
 */
public final class ResearchZfpCliBench {

    private static final String ROW_REVERSIBLE = "ext_zfp_zfpy";
    private static final double[] TOLERANCES = {1e-1, 1e-2, 1e-3};
    private static final String[] ROW_TOLERANCE = {
        "ext_zfp_tol_1e-1", "ext_zfp_tol_1e-2", "ext_zfp_tol_1e-3"
    };

    private ResearchZfpCliBench() {}

    public static void accumulateRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runZfpRows64(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            CSVPrinter csvWriter)
            throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runZfpRows64(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    private static void runZfpRows64(
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
        ArrayList<ArrayList<Double>> columns =
                BenchmarkColumns.doubleColumnsForCodecBench(tsType, intCols, floatCols);
        if (columns.isEmpty()) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File zfp = resolveZfpExecutable(root);
        if (zfp == null) {
            return;
        }
        runAllZfpRows(columns, zfp, totals, csvWriter, skipAlgorithms);
    }

    private static void runAllZfpRows(
            List<ArrayList<Double>> columns,
            File zfp,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms)
            throws IOException {
        if (!NativeLzFamilyBench.algorithms().contains(ROW_REVERSIBLE)
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW_REVERSIBLE)) {
            long[] t = new long[4];
            double[] loss = new double[2];
            if (benchZfpDoubleColumns(columns, zfp, null, t, loss) && t[0] > 0) {
                recordRow(ROW_REVERSIBLE, t, loss, totals, csvWriter);
            }
        }
        for (int i = 0; i < TOLERANCES.length; i++) {
            String row = ROW_TOLERANCE[i];
            if (AlgorithmAllowList.shouldSkip(skipAlgorithms, row)) {
                continue;
            }
            long[] t = new long[4];
            double[] loss = new double[2];
            if (benchZfpDoubleColumns(columns, zfp, TOLERANCES[i], t, loss) && t[0] > 0) {
                recordRow(row, t, loss, totals, csvWriter);
            }
        }
    }

    private static void recordRow(
            String row,
            long[] t,
            double[] loss,
            BenchmarkTotals totals,
            CSVPrinter csvWriter)
            throws IOException {
        if (totals != null) {
            totals.merge(row, t, loss[0], loss[1]);
        } else if (csvWriter != null) {
            csvWriter.printRecord(row, t[0], t[1], t[2], t[3]);
        }
    }

    static boolean benchEnabled() {
        String flag = System.getenv("WEB_COMPRESSION_ZFP_BENCH");
        return flag == null || flag.isEmpty() || !"0".equals(flag.trim());
    }

    static File resolveZfpExecutable(File root) {
        File env = NativeCliExecutable.fromEnv("WEB_COMPRESSION_ZFP");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root,
                "external/compression/zfp/build/bin/zfp",
                "external/compression/zfp/build/Release/zfp",
                "cpp/build/ext_zfp/bin/zfp",
                "cpp/build/ext_zfp/Release/zfp");
    }

    static boolean benchZfpDoubleColumns(
            List<ArrayList<Double>> columns,
            File zfp,
            Double tolerance,
            long[] out,
            double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_zfp");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        final boolean reversible = tolerance == null;
        final double verifyAtol =
                reversible ? 0.0 : Math.max(tolerance * 16.0, 1e-6);
        try {
            for (List<Double> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 8];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                ArrayList<Double> origDoubles = new ArrayList<>(n);
                for (Double dv : col) {
                    double d = dv == null ? Double.NaN : dv;
                    bb.putDouble(d);
                    origDoubles.add(d);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path zfpOut = Path.of(in + ".zfp");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(zfpOut);
                    Files.deleteIfExists(dec);

                    List<String> encCmd = new ArrayList<>();
                    encCmd.add(zfp.getAbsolutePath());
                    encCmd.add("-i");
                    encCmd.add(in.toAbsolutePath().toString());
                    encCmd.add("-z");
                    encCmd.add(zfpOut.toAbsolutePath().toString());
                    encCmd.add("-d");
                    encCmd.add("-1");
                    encCmd.add(Integer.toString(n));
                    if (reversible) {
                        encCmd.add("-R");
                    } else {
                        encCmd.add("-a");
                        encCmd.add(formatTolerance(tolerance));
                    }
                    encCmd.add("-h");
                    encCmd.add("-q");

                    long t0 = System.nanoTime();
                    ProcessBuilder enc = new ProcessBuilder(encCmd);
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
                    if (!Files.isRegularFile(zfpOut)) {
                        continue;
                    }
                    long csize = Files.size(zfpOut);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    zfp.getAbsolutePath(),
                                    "-z",
                                    zfpOut.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString(),
                                    "-h",
                                    "-q");
                    decPb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    decPb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = decPb.start();
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

                    if (!Files.isRegularFile(dec) || Files.size(dec) != raw.length) {
                        continue;
                    }
                    byte[] back = Files.readAllBytes(dec);
                    if (reversible) {
                        if (!java.util.Arrays.equals(raw, back)) {
                            continue;
                        }
                    } else if (!doubleWithinTolerance(raw, back, verifyAtol)) {
                        continue;
                    }

                    ArrayList<Double> decDoubles = doubleBytesToList(back);
                    BenchmarkLoss.accumulateDoubleListLoss(origDoubles, decDoubles, lossAcc);

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(zfpOut);
                        Files.deleteIfExists(dec);
                    } catch (Exception ignored) {
                    }
                }
            }
        } catch (Exception ignored) {
            out[0] = 0;
            return false;
        } finally {
            deleteRecursiveQuiet(tmpDir);
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    private static ArrayList<Double> doubleBytesToList(byte[] raw) {
        int n = raw.length / 8;
        ArrayList<Double> out = new ArrayList<>(n);
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            out.add(bb.getDouble());
        }
        return out;
    }

    private static boolean doubleWithinTolerance(byte[] orig, byte[] dec, double atol) {
        if (orig.length != dec.length) {
            return false;
        }
        ByteBuffer ob = ByteBuffer.wrap(orig).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer db = ByteBuffer.wrap(dec).order(ByteOrder.LITTLE_ENDIAN);
        int n = orig.length / 8;
        for (int i = 0; i < n; i++) {
            double o = ob.getDouble();
            double d = db.getDouble();
            if (Double.isNaN(o) && Double.isNaN(d)) {
                continue;
            }
            if (Math.abs(o - d) > atol) {
                return false;
            }
        }
        return true;
    }

    private static String formatTolerance(double tolerance) {
        return String.format(Locale.US, "%g", tolerance);
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
