#pragma once

#include <string>

bool run_java_companion_phase(const std::string &repoRoot, const std::string &inputCsvPath,
                              const std::string &dataType, const std::string &appendReportPath,
                              const char *phase /* "external-only"; TSFile rows are native in benchmark_cpp */);
