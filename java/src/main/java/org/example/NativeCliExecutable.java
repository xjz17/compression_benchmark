package org.example;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Resolve native CLI tools built under {@code external/compression/}. On Windows, CMake/MinGW
 * outputs {@code fpzip.exe} / {@code sz3.exe} while Unix builds use extensionless names.
 */
public final class NativeCliExecutable {

    private NativeCliExecutable() {}

    /**
     * Windows SZ3/fpzip CLIs often depend on DLLs next to the binary or under
     * {@code SZ3/build/tools/zstd/libzstd.dll}. Prepend those dirs to {@code PATH} before
     * {@link ProcessBuilder#start()}.
     */
    public static void configureNativeCliProcess(ProcessBuilder pb, File cliExecutable, File repoRoot) {
        if (cliExecutable == null) {
            return;
        }
        List<String> prepend = new ArrayList<>();
        File parent = cliExecutable.getParentFile();
        if (parent != null) {
            prepend.add(parent.getAbsolutePath());
        }
        if (repoRoot != null) {
            File zstdDir = new File(repoRoot, "external/compression/SZ3/build/tools/zstd");
            if (zstdDir.isDirectory()) {
                prepend.add(zstdDir.getAbsolutePath());
            }
        }
        String mingwBin = System.getenv("COMPRESSION_BENCHMARK_MINGW_BIN");
        if (mingwBin != null && !mingwBin.isBlank()) {
            prepend.add(new File(mingwBin.trim()).getAbsolutePath());
        }
        String extra = System.getenv("COMPRESSION_BENCHMARK_NATIVE_CLI_PATH");
        if (extra != null && !extra.isBlank()) {
            for (String part : extra.split(File.pathSeparator)) {
                if (part != null && !part.isBlank()) {
                    prepend.add(part.trim());
                }
            }
        }
        if (!prepend.isEmpty()) {
            String pathKey = pathEnvKey();
            Map<String, String> env = pb.environment();
            String cur = env.get(pathKey);
            if (cur == null) {
                cur = System.getenv(pathKey);
            }
            StringBuilder merged = new StringBuilder();
            for (int i = 0; i < prepend.size(); i++) {
                if (i > 0) {
                    merged.append(File.pathSeparator);
                }
                merged.append(prepend.get(i));
            }
            if (cur != null && !cur.isEmpty()) {
                merged.append(File.pathSeparator).append(cur);
            }
            env.put(pathKey, merged.toString());
        }
        if (parent != null) {
            pb.directory(parent);
        }
    }

    private static String pathEnvKey() {
        return isWindows() ? "Path" : "PATH";
    }

    public static File fromEnv(String envVar) {
        String env = System.getenv(envVar);
        if (env == null || env.isBlank()) {
            return null;
        }
        return firstRunnable(new File(env.trim()));
    }

    /** First existing file among {@code root/relative} paths (tries {@code .exe} when needed). */
    public static File firstRunnable(File root, String... relativePaths) {
        if (root == null || relativePaths == null) {
            return null;
        }
        for (String rel : relativePaths) {
            if (rel == null || rel.isEmpty()) {
                continue;
            }
            File hit = firstRunnable(new File(root, rel));
            if (hit != null) {
                return hit;
            }
        }
        return null;
    }

    public static File firstRunnable(File path) {
        if (path == null) {
            return null;
        }
        File hit = runnableFile(path);
        if (hit != null) {
            return hit;
        }
        String name = path.getName();
        if (!name.isEmpty() && !name.toLowerCase().endsWith(".exe")) {
            return runnableFile(new File(path.getParentFile(), name + ".exe"));
        }
        return null;
    }

    private static File runnableFile(File file) {
        if (file != null && file.isFile() && file.canExecute()) {
            return file.getAbsoluteFile();
        }
        return null;
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "");
        return os.toLowerCase().contains("win");
    }
}
