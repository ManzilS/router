"""Root conftest — makes `src` importable and provides shared fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))
