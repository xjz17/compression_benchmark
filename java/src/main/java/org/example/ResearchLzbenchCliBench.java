package org.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Disabled legacy lzbench byte-codec rows via {@code scripts/lzbench_patch_cli.py}.
 *
 * <p>Catalog byte-codec rows are produced by {@link NativeLzFamilyBench}; this class remains only
 * so old references compile and never launches the external subprocess.
 */
public final class ResearchLzbenchCliBench {

    private ResearchLzbenchCliBench() {}

    public static void accumulateByteColumns(
            List<byte[]> columnByteData,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        if (totals == null || columnByteData == null || columnByteData.isEmpty() || !benchEnabled()) {
            return;
        }
        if (!AlgorithmAllowList.allowlistMayWantLzbench(skipAlgorithms)) {
            return;
        }
        File root = ResearchNativeCliBench.repoRootFromUserDir();
        File script = resolvePatchCliScript(root);
        if (script == null) {
            return;
        }
        String python = resolvePython();
        for (byte[] data : columnByteData) {
            accumulatePatchBytes(python, script, root, data, totals, skipAlgorithms);
        }
    }

    private static void accumulatePatchBytes(
            String python,
            File script,
            File repoRoot,
            byte[] payload,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        if (payload == null || payload.length == 0) {
            return;
        }
        for (String[] fields : runPatchCli(python, script, repoRoot, payload)) {
            if (fields.length < 5) {
                continue;
            }
            String algo = fields[0].trim();
            if (algo.isEmpty() || AlgorithmAllowList.shouldSkip(skipAlgorithms, algo)) {
                continue;
            }
            try {
                long orig = Long.parseLong(fields[1].trim());
                long comp = Long.parseLong(fields[2].trim());
                long ct = Long.parseLong(fields[3].trim());
                long dt = Long.parseLong(fields[4].trim());
                if (orig > 0 && dt > 0) {
                    totals.merge(algo, orig, comp, ct, dt);
                }
            } catch (NumberFormatException ignored) {
                // skip row
            }
        }
    }

    private static boolean benchEnabled() {
        // lzbench CLI is disabled; catalog rows are produced by NativeLzFamilyBench.
        return false;
    }

    private static String resolvePython() {
        String py = System.getenv("WEB_COMPRESSION_PYTHON");
        if (py != null && !py.isBlank()) {
            return py.trim();
        }
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win") ? "python" : "python3";
    }

    static File resolvePatchCliScript(File root) {
        if (root == null) {
            return null;
        }
        File script = new File(root, "scripts/lzbench_patch_cli.py");
        return script.isFile() ? script : null;
    }

    private static List<String[]> runPatchCli(
            String python, File script, File repoRoot, byte[] payload) throws IOException {
        Path tmp = Files.createTempFile("wc_lzbench_patch_", ".bin");
        try {
            Files.write(tmp, payload);
            ProcessBuilder pb =
                    new ProcessBuilder(
                            python,
                            script.getAbsolutePath(),
                            tmp.toAbsolutePath().toString());
            if (repoRoot != null) {
                pb.environment().put("WEB_COMPRESSION_REPO_ROOT", repoRoot.getAbsolutePath());
            }
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            List<String[]> rows = new ArrayList<>();
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;
                    }
                    String[] parts = line.split(",", -1);
                    if (parts.length >= 5) {
                        rows.add(parts);
                    }
                }
            }
            try {
                proc.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return rows;
        } finally {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException ignored) {
                // best effort
            }
        }
    }
}
