"""Backward-compatible shim for scripts importing ``home_leaderboard_boxplot_data``."""

from myapp.home_leaderboard_boxplot_data import (  # noqa: F401
    MIN_RATE_FOR_BOX_PLOT,
    boxplot_group_label as catalog_label,
    build_home_leaderboard_boxplot_groups,
    collect_metrics,
)
