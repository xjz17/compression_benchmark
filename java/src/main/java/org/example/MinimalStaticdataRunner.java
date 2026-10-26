package org.example;

/**
 * Tiny entrypoint for rerunning one CSV / one column through the existing patch benchmark.
 */
public final class MinimalStaticdataRunner {
    private MinimalStaticdataRunner() {}

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: MinimalStaticdataRunner <csvPath> <dataType> <outputCsv> <columnInfoPath> [columnIndex]");
            System.exit(2);
        }
        String csvPath = args[0];
        String dataType = args[1];
        String outputCsv = args[2];
        String columnInfoPath = args[3];
        Integer columnIndex = null;
        if (args.length >= 5 && args[4] != null && !args[4].isBlank()) {
            columnIndex = Integer.valueOf(args[4].trim());
        }
        ProcessorPatchBench.run(csvPath, dataType, outputCsv, columnInfoPath, columnIndex);
    }
}
