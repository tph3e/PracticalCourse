from __future__ import annotations

import sys
from pathlib import Path


JOAO_ROOT = Path(__file__).resolve().parents[1]
if str(JOAO_ROOT) not in sys.path:
    sys.path.insert(0, str(JOAO_ROOT))

PROJECT_ROOT = JOAO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
