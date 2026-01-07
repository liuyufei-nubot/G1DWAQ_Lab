from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ['PROJECT_ROOT']