"""Robot configuration modules for frog_lab."""

from __future__ import annotations

import os

# Convenience paths for local assets.
FROG_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
FROG_LAB_DATA_DIR = os.path.join(FROG_LAB_EXT_DIR, "model")
