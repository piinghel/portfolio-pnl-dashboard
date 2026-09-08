"""Pure visual configuration for the dashboard instance."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ChartSettings:
    """Visual choices supplied by the dashboard instance, independent of data."""

    line_height: int = 360
    pnl_drawdown_height: int = 520
    stock_history_height: int = 560
    stock_guide_limit: int = 8
    contribution_height: int = 620
    contribution_max_periods: int = 24
    contribution_risk_window: int = 63
    bar_row_height: int = 25
    font_size: int = 14
    label_length: int = 28
    pnl_decimals: int = 3

    def __post_init__(self) -> None:
        limits = {
            "line_height": (240, 800),
            "contribution_height": (500, 1000),
            "contribution_max_periods": (12, 60),
            "contribution_risk_window": (21, 252),
            "pnl_drawdown_height": (400, 1000),
            "stock_history_height": (500, 1000),
            "stock_guide_limit": (0, 100),
            "bar_row_height": (18, 48),
            "font_size": (12, 22),
            "label_length": (16, 50),
            "pnl_decimals": (2, 3),
        }
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer from {lower} to {upper}")


DEFAULT_CHARTS = ChartSettings()
