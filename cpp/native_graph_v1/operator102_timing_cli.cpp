// Build the production operator102 column loader/dispatcher as an isolated
// timing-only executable.  The macro selects a distinct wire schema and skips
// entropy/Zstd-ablation work; the accepted measurement binary is unchanged.
#define OPERATOR102_TIMING_MODE 1
#include "operator102_measure_cli.cpp"
