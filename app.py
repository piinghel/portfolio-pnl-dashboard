"""Run the public demo with `streamlit run app.py`."""

from __future__ import annotations

from pathlib import Path

import attribution_dashboard.realized_page as page

page.render_config(Path(__file__).with_name("config.yaml"))
