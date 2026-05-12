"""
Proje kökünü sys.path'e ekler; betikler scripts/ altından çalıştırıldığında
`config`, `asr` vb. paket içe aktarımları çalışır.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ROOT_STR = str(PROJECT_ROOT)

if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)
