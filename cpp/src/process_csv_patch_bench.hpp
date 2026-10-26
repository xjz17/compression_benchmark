#pragma once

#include <string>

/** Catalog staticdata: patch CSV rows, accumulate metrics, write report (uses column_info max points). */
void processCsvSimplePatch(const std::string& csvPath, const std::string& dataType,
                             const std::string& outputCsvPath,
                             const std::string& columnInfoPath,
                             int columnIndex = -1);
