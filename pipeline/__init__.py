"""Maintained scientific workflows; local work products belong under tmp/."""
import sys
from pathlib import Path

# Route imported-module bytecode to the same work root as all other caches.
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(Path(__file__).resolve().parents[1] / 'tmp/cache/python')
