import sys
from pathlib import Path

COGWHEEL_PATH = Path(__file__).parents[2].resolve()
sys.path.append(COGWHEEL_PATH.as_posix())
