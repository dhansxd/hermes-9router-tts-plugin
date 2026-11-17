"""Run the offline regression suite using the Hermes Python environment."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tests" / "test_regressions.py"), run_name="__main__")
