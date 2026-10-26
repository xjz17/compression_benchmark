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
 * Disabled legacy Blosc2 CLI helper. Runtime {@code blosc2_blosclz} rows are emitted by
 * {@link NativeLzFamilyBench} so Java benchmark runs do not launch {@code blosc2_col_bench}.
 */
public final class ResearchBlosc2CliBench {

    public static final String ROW = "blosc2_blosclz";

    private ResearchBlosc2CliBench() {}

    public static void accumulateRows64(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        return;
    }

    public static void accumulateByteColumns(
            List<byte[]> columnByteData,
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

    private static void runRows64(
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
        final String flag = System.getenv("COMPRESSION_BENCHMARK_BLOSC2_BENCH");
        final boolean bench = flag == null || flag.isEmpty() || "1".equals(flag.trim());
        if (!bench) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File cli = resolveBlosc2ColBenchExecutable(root);
        if (cli == null) {
            return;
        }
        long[] t = new long[4];
        double[] loss = new double[2];
        boolean ok = false;
        if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
            ok = benchFloat32Columns(floatCols, cli, t, loss);
        } else if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
            ok = benchInt64Columns(intCols, cli, t, loss);
        }
        if (ok && t[0] > 0) {
            if (totals != null) {
                totals.merge(ROW, t, loss[0], loss[1]);
            } else if (csvWriter != null) {
                csvWriter.printRecord(ROW, t[0], t[1], t[2], t[3]);
            }
        }
    }

    static File resolveBlosc2ColBenchExecutable(File root) {
        File env = NativeCliExecutable.fromEnv("COMPRESSION_BENCHMARK_BLOSC2_COL_BENCH");
        if (env != null) {
            return env;
        }
        return NativeCliExecutable.firstRunnable(
                root,
                "external/compression/c-blosc2/build/bin/blosc2_col_bench");
    }

    static boolean benchFloat32Columns(
            List<? extends List<Double>> columns, File cli, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_blosc2_f");
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
                byte[] raw = new byte[n * 4];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                ArrayList<Double> origDoubles = new ArrayList<>(n);
                for (Double dv : col) {
                    float f = dv == null ? Float.NaN : dv.floatValue();
                    bb.putFloat(f);
                    origDoubles.add((double) f);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path b2 = Path.of(in + ".b2");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(b2);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-c",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    b2.toAbsolutePath().toString(),
                                    "-s",
                                    "4");
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    if (!p1.waitFor(120, TimeUnit.SECONDS) || p1.exitValue() != 0) {
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(b2)) {
                        continue;
                    }
                    long csize = Files.size(b2);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-d",
                                    "-i",
                                    b2.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
                    decPb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    decPb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = decPb.start();
                    if (!p2.waitFor(120, TimeUnit.SECONDS) || p2.exitValue() != 0) {
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    byte[] back = Files.readAllBytes(dec);
                    if (back.length != raw.length || !java.util.Arrays.equals(back, raw)) {
                        continue;
                    }
                    ArrayList<Double> decDoubles = float32BytesToDoubleList(back);
                    BenchmarkLoss.accumulateDoubleListLoss(origDoubles, decDoubles, lossAcc);

                    orig += (long) n * 4;
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(b2);
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

    static boolean benchInt64Columns(
            List<? extends List<Long>> columns, File cli, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_blosc2_i");
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
                ArrayList<Double> origDoubles = new ArrayList<>(n);
                for (Long lv : col) {
                    long v = lv == null ? 0L : lv;
                    bb.putLong(v);
                    origDoubles.add((double) v);
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path b2 = Path.of(in + ".b2");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(b2);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-c",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    b2.toAbsolutePath().toString(),
                                    "-s",
                                    "8");
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    if (!p1.waitFor(120, TimeUnit.SECONDS) || p1.exitValue() != 0) {
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(b2)) {
                        continue;
                    }
                    long csize = Files.size(b2);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-d",
                                    "-i",
                                    b2.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
                    decPb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    decPb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = decPb.start();
                    if (!p2.waitFor(120, TimeUnit.SECONDS) || p2.exitValue() != 0) {
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    byte[] back = Files.readAllBytes(dec);
                    if (back.length != raw.length || !java.util.Arrays.equals(back, raw)) {
                        continue;
                    }
                    ArrayList<Double> decDoubles = int64BytesToDoubleList(back);
                    BenchmarkLoss.accumulateDoubleListLoss(origDoubles, decDoubles, lossAcc);

                    orig += (long) n * 8;
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(b2);
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

    static boolean benchByteColumns(List<byte[]> columns, File cli, long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_blosc2_s");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        try {
            for (byte[] raw : columns) {
                if (raw == null || raw.length < 1) {
                    continue;
                }
                Path in = tmpDir.resolve("col-" + System.nanoTime() + ".bin");
                Path b2 = Path.of(in + ".b2");
                Path dec = Path.of(in + ".dec.bin");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(b2);
                    Files.deleteIfExists(dec);

                    long t0 = System.nanoTime();
                    ProcessBuilder enc =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-c",
                                    "-i",
                                    in.toAbsolutePath().toString(),
                                    "-o",
                                    b2.toAbsolutePath().toString(),
                                    "-s",
                                    "1");
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    if (!p1.waitFor(120, TimeUnit.SECONDS) || p1.exitValue() != 0) {
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(b2)) {
                        continue;
                    }
                    long csize = Files.size(b2);

                    long t2 = System.nanoTime();
                    ProcessBuilder decPb =
                            new ProcessBuilder(
                                    cli.getAbsolutePath(),
                                    "-d",
                                    "-i",
                                    b2.toAbsolutePath().toString(),
                                    "-o",
                                    dec.toAbsolutePath().toString());
                    decPb.redirectError(ProcessBuilder.Redirect.DISCARD);
                    decPb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = decPb.start();
                    if (!p2.waitFor(120, TimeUnit.SECONDS) || p2.exitValue() != 0) {
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    byte[] back = Files.readAllBytes(dec);
                    if (back.length != raw.length || !java.util.Arrays.equals(back, raw)) {
                        continue;
                    }

                    orig += raw.length;
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(b2);
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

    private static ArrayList<Double> float32BytesToDoubleList(byte[] raw) {
        ArrayList<Double> out = new ArrayList<>();
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        while (bb.remaining() >= 4) {
            out.add((double) bb.getFloat());
        }
        return out;
    }

    private static ArrayList<Double> int64BytesToDoubleList(byte[] raw) {
        ArrayList<Double> out = new ArrayList<>();
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        while (bb.remaining() >= 8) {
            out.add((double) bb.getLong());
        }
        return out;
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
