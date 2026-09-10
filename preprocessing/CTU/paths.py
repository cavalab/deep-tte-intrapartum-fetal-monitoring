"""Shared default paths for the CTU-UHB preprocessing workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CTU_DIR = DATA_DIR / "CTU"
CTU_RAW_DIR = CTU_DIR / "raw"
CTU_WFDB_DIR = CTU_RAW_DIR / "physionet.org" / "files" / "ctu-uhb-ctgdb" / "1.0.0"
CTU_CONVERTED_DIR = CTU_DIR / "converted"
PROCESSED_DATA_DIR = DATA_DIR

