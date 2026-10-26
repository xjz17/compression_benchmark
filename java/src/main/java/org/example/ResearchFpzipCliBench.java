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
 * Disabled legacy fpzip CLI helper. Runtime {@code fpzip} rows are emitted by
 * {@link NativeLzFamilyBench} so Java benchmark runs do not launch LLNL fpzip.
 */
public final class ResearchFpzipCliBench {

    private static final String ROW = "fpzip";

    private ResearchFpzipCliBench() {}

    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        return;
    }

    public static void printRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            CSVPrinter csvWriter)
            throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        return;
    }

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
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        return;
    }

    private static void runFpzipRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms)
            throws IOException {
        if (AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW)) {
            return;
        }
        final String flag = System.getenv("WEB_COMPRESSION_FPZIP_BENCH");
        final boolean bench = flag == null || flag.isEmpty() || "1".equals(flag.trim());
        if (!bench) {
            return;
        }
        ArrayList<ArrayList<Float>> columns =
                BenchmarkColumns.floatColumnsForCodecBench(tsType, intCols, floatCols);
        if (columns.isEmpty()) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File fpzip = resolveFpzipExecutable(root);
        if (fpzip == null) {
            return;
        }
        long[] t = new long[4];
        double[] loss = new double[2];
        if (benchFpzipColumns(columns, fpzip, t, loss) && t[0] > 0) {
            if (totals != null) {
                totals.merge(ROW, t, loss[0], loss[1]);
            } else if (csvWriter != null) {
                csvWriter.printRecord(ROW, t[0], t[1], t[2], t[3]);
            }
        }
    }

    private static void runFpzipRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms)
            throws IOException {
        if (AlgorithmAllowList.shouldSkip(skipAlgorithms, ROW)) {
            return;
        }
        final String flag = System.getenv("WEB_COMPRESSION_FPZIP_BENCH");
        final boolean bench = flag == null || flag.isEmpty() || "1".equals(flag.trim());
        if (!bench) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File fpzip = resolveFpzipExecutable(root);
        if (fpzip == null) {
            return;
        }
        long[] t = new long[4];
        double[] loss = new double[2];
        boolean ok = false;
        if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
            ok = benchFpzipColumnsDouble(floatCols, fpzip, t, loss);
        } else if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
            ok = benchFpzipColumnsDouble(BenchmarkColumns.integersAsDoubleColumns(intCols), fpzip, t, loss);
        }
        if (ok && t[0] > 0) {
            if (totals != null) {
                totals.merge(ROW, t, loss[0], loss[1]);
            } else if (csvWriter != null) {
                csvWriter.printRecord(ROW, t[0], t[1], t[2], t[3]);
            }
        }
    }

    static File resolveFpzipExecutable(File root) {
        File env = NativeCliExecutable.fromEnv("WEB_COMPRESSION_FPZIP");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root,
                "external/compression/fpzip/build/bin/fpzip",
                "external/compression/fpzip/build/utils/fpzip",
                "external/compression/fpzip/bin/fpzip",
                "cpp/build/ext_fpzip/bin/fpzip",
                "cpp/build/ext_fpzip/utils/fpzip");
    }

    static boolean benchFpzipColumns(List<ArrayList<Float>> columns, File fpzip, long[] out) {
        return benchFpzipColumns(columns, fpzip, out, null);
    }

    static boolean benchFpzipColumns(
            List<ArrayList<Float>> columns, File fpzip, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_fpzip");
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
                byte[] raw = new byte[n * 4];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : col) {
                    bb.putFloat(f);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path fpz = Path.of(in + ".fpz");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(fpz);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    fpz.toAbsolutePath().toString(),
                                    "-1",
                                    Integer.toString(n));
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
                    if (!Files.isRegularFile(fpz)) {
                        continue;
                    }
                    long csize = Files.size(fpz);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-d",
                                    "-i",
                                    fpz.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
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

                    if (Files.isRegularFile(dec) && Files.size(dec) == raw.length) {
                        byte[] back = Files.readAllBytes(dec);
                        BenchmarkLoss.accumulateByteLoss(raw, back, lossAcc);
                    }

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(fpz);
                        Files.deleteIfExists(dec);
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
        return orig > 0;
    }

    static boolean benchFpzipColumnsDouble(List<? extends List<Double>> columns, File fpzip, long[] out) {
        return benchFpzipColumnsDouble(columns, fpzip, out, null);
    }

    static boolean benchFpzipColumnsDouble(
            List<? extends List<Double>> columns, File fpzip, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_fpzip64");
        } catch (Exception e) {
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
                for (Double dv : col) {
                    bb.putDouble(dv == null ? Double.NaN : dv);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path fpz = Path.of(in + ".fpz");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(fpz);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-t",
                                    "double",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    fpz.toAbsolutePath().toString(),
                                    "-1",
                                    Integer.toString(n));
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
                    if (!Files.isRegularFile(fpz)) {
                        continue;
                    }
                    long csize = Files.size(fpz);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-d",
                                    "-i",
                                    fpz.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
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
                    byte[] decRaw = Files.readAllBytes(dec);
                    if (decRaw.length != raw.length || !java.util.Arrays.equals(decRaw, raw)) {
                        continue;
                    }
                    BenchmarkLoss.accumulateByteLoss(raw, decRaw, lossAcc);
                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(fpz);
                        Files.deleteIfExists(dec);
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
        return orig > 0;
    }

    static boolean benchFpzipColumnsLong(List<? extends List<Long>> columns, File fpzip, long[] out) {
        return benchFpzipColumnsLong(columns, fpzip, out, null);
    }

    static boolean benchFpzipColumnsLong(
            List<? extends List<Long>> columns, File fpzip, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_fpzip64i");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        try {
            for (List<Long> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 8];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (Long lv : col) {
                    bb.putLong(lv == null ? 0L : lv);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path fpz = Path.of(in + ".fpz");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(fpz);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-t",
                                    "double",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    fpz.toAbsolutePath().toString(),
                                    "-1",
                                    Integer.toString(n));
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
                    if (!Files.isRegularFile(fpz)) {
                        continue;
                    }
                    long csize = Files.size(fpz);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    fpzip.getAbsolutePath(),
                                    "-q",
                                    "-d",
                                    "-i",
                                    fpz.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
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
                    byte[] decRaw = Files.readAllBytes(dec);
                    if (decRaw.length != raw.length || !java.util.Arrays.equals(decRaw, raw)) {
                        continue;
                    }
                    BenchmarkLoss.accumulateByteLoss(raw, decRaw, lossAcc);
                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(fpz);
                        Files.deleteIfExists(dec);
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
        return orig > 0;
    }

    private static void deleteRecursive(Path root) throws IOException {
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
    }
}
