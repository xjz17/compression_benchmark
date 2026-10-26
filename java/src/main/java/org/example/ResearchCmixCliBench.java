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
 * cmix lossless CLI ({@code cmix -c} / {@code cmix -d}) on raw float32 column bytes.
 *
 * <p>Binary: {@code external/compression/cmix/cmix} after {@code make}, or {@code WEB_COMPRESSION_CMIX}.
 * cmix is CPU-only (no GPU). Opt-in: {@code WEB_COMPRESSION_CMIX_BENCH=1}. Optional
 * {@code WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES} limits each column prefix. Set
 * {@code WEB_COMPRESSION_CMIX_NO_PREPROCESS=1} to use {@code -n} instead of {@code -c} (faster staticdata).
 */
public final class ResearchCmixCliBench {

    private ResearchCmixCliBench() {}

    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        runCmixRows(tsType, intCols, floatCols, totals, null, skipAlgorithms);
    }

    public static void printRows(
            TSDataType tsType,
            @SuppressWarnings("unused") ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        runCmixRows(tsType, intCols, floatCols, null, csvWriter, skipAlgorithms);
    }

    private static void runCmixRows(
            TSDataType tsType,
            ArrayList<ArrayList<Integer>> intCols,
            ArrayList<ArrayList<Float>> floatCols,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        if (!"1".equals(System.getenv("WEB_COMPRESSION_CMIX_BENCH"))) {
            return;
        }
        ArrayList<ArrayList<Float>> columns =
                BenchmarkColumns.floatColumnsForCodecBench(tsType, intCols, floatCols);
        if (columns.isEmpty()) {
            return;
        }
        if (AlgorithmAllowList.shouldSkip(skipAlgorithms, "ext_cmix")) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File cmix = resolveCmixExecutable(root);
        if (cmix == null) {
            return;
        }
        int timeoutSec = cmixTimeoutSec();
        long[] t = new long[4];
        double[] loss = new double[2];
        if (benchCmixFloat(columns, cmix, timeoutSec, t, loss) && t[0] > 0) {
            if (totals != null) {
                totals.merge("ext_cmix", t, loss[0], loss[1]);
            } else if (csvWriter != null) {
                csvWriter.printRecord("ext_cmix", t[0], t[1], t[2], t[3]);
            }
        }
    }

    static File resolveCmixExecutable(File root) {
        String env = System.getenv("WEB_COMPRESSION_CMIX");
        if (env != null && !env.isEmpty()) {
            File f = new File(env);
            if (f.isFile() && f.canExecute()) {
                return f;
            }
        }
        File cand = new File(root, "external/compression/cmix/cmix");
        if (cand.isFile() && cand.canExecute()) {
            return cand;
        }
        return null;
    }

    static int cmixTimeoutSec() {
        String e = System.getenv("WEB_COMPRESSION_CMIX_TIMEOUT_SEC");
        if (e == null || e.isEmpty()) {
            return 600;
        }
        try {
            int v = Integer.parseInt(e.trim());
            if (v <= 0) {
                return 0;
            }
            return Math.min(v, 86400);
        } catch (NumberFormatException ex) {
            return 600;
        }
    }

    /**
     * If {@code WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES} is set to a positive value, only the leading
     * prefix of each float column (that many bytes, multiple of 4) is compressed — avoids multi-hour
     * staticdata runs. {@code 0} or unset = use full column.
     */
    /** {@code true} → compress with {@code -n} (no preprocessor); else {@code -c}. */
    static boolean cmixNoPreprocess() {
        return "1".equals(System.getenv("WEB_COMPRESSION_CMIX_NO_PREPROCESS"));
    }

    static int cmixMaxColumnBytes() {
        String e = System.getenv("WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES");
        if (e == null || e.isEmpty()) {
            return 0;
        }
        try {
            int v = Integer.parseInt(e.trim());
            if (v <= 0) {
                return 0;
            }
            return Math.min(v, 256 * 1024 * 1024);
        } catch (NumberFormatException ex) {
            return 0;
        }
    }

    static boolean benchCmixFloat(
            List<ArrayList<Float>> columns, File cmix, int timeoutSec, long[] out) {
        return benchCmixFloat(columns, cmix, timeoutSec, out, null);
    }

    static boolean benchCmixFloat(
            List<ArrayList<Float>> columns, File cmix, int timeoutSec, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        Path tmpDir;
        try {
            tmpDir = Files.createTempDirectory("wc_cmix_float");
        } catch (Exception e) {
            out[0] = 0;
            return false;
        }
        int maxColBytes = cmixMaxColumnBytes();
        try {
            for (List<Float> col : columns) {
                int n = col.size();
                if (n < 1) {
                    continue;
                }
                if (maxColBytes > 0) {
                    int maxFloats = maxColBytes / 4;
                    if (n > maxFloats) {
                        n = maxFloats;
                    }
                }
                if (n < 1) {
                    continue;
                }
                byte[] raw = new byte[n * 4];
                ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < n; i++) {
                    bb.putFloat(col.get(i));
                }
                Path in = tmpDir.resolve("in-" + System.nanoTime() + ".dat");
                Path fcomp = tmpDir.resolve("cmp-" + System.nanoTime() + ".cmix");
                Path fdec = tmpDir.resolve("dec-" + System.nanoTime() + ".dat");
                try {
                    Files.write(in, raw);
                    Files.deleteIfExists(fcomp);
                    Files.deleteIfExists(fdec);

                    List<String> encCmd = new ArrayList<>();
                    encCmd.add(cmix.getAbsolutePath());
                    encCmd.add(cmixNoPreprocess() ? "-n" : "-c");
                    encCmd.add(in.toAbsolutePath().toString());
                    encCmd.add(fcomp.toAbsolutePath().toString());

                    long t0 = System.nanoTime();
                    ProcessBuilder enc = new ProcessBuilder(encCmd);
                    enc.redirectError(ProcessBuilder.Redirect.DISCARD);
                    enc.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p1 = enc.start();
                    try {
                        boolean ok =
                                timeoutSec <= 0
                                        ? (p1.waitFor() == 0)
                                        : p1.waitFor(timeoutSec, TimeUnit.SECONDS)
                                                && p1.exitValue() == 0;
                        if (!ok) {
                            p1.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p1.destroyForcibly();
                        continue;
                    }
                    long t1 = System.nanoTime();
                    if (!Files.isRegularFile(fcomp)) {
                        continue;
                    }
                    long csize = Files.size(fcomp);

                    long t2 = System.nanoTime();
                    ProcessBuilder dec =
                            new ProcessBuilder(
                                    cmix.getAbsolutePath(),
                                    "-d",
                                    fcomp.toAbsolutePath().toString(),
                                    fdec.toAbsolutePath().toString());
                    dec.redirectError(ProcessBuilder.Redirect.DISCARD);
                    dec.redirectOutput(ProcessBuilder.Redirect.DISCARD);
                    Process p2 = dec.start();
                    try {
                        boolean ok2 =
                                timeoutSec <= 0
                                        ? (p2.waitFor() == 0)
                                        : p2.waitFor(timeoutSec, TimeUnit.SECONDS)
                                                && p2.exitValue() == 0;
                        if (!ok2) {
                            p2.destroyForcibly();
                            continue;
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        p2.destroyForcibly();
                        continue;
                    }
                    long t3 = System.nanoTime();

                    if (!Files.isRegularFile(fdec) || Files.size(fdec) != raw.length) {
                        continue;
                    }
                    byte[] back = Files.readAllBytes(fdec);
                    if (!java.util.Arrays.equals(back, raw)) {
                        continue;
                    }
                    BenchmarkLoss.accumulateByteLoss(raw, back, lossAcc);

                    orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                    comp += csize;
                    ct += t1 - t0;
                    dt += t3 - t2;
                } catch (Exception ignored) {
                    continue;
                } finally {
                    try {
                        Files.deleteIfExists(in);
                        Files.deleteIfExists(fcomp);
                        Files.deleteIfExists(fdec);
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
