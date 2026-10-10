from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_FILE = DATA_DIR / "raw" / "dataset_TSMC2014_NYC.txt"

COLUMN_NAMES = [
    "user_id",
    "venue_id",
    "category_id",
    "category_name",
    "latitude",
    "longitude",
    "timezone_offset",
    "utc_timestamp",
]


def load_raw_data(filepath=None):
    target = Path(filepath) if filepath else RAW_FILE
    if not target.exists():
        raise FileNotFoundError(f"Dataset not found at {target}")
    return pd.read_csv(target, sep="\t", header=None, names=COLUMN_NAMES, encoding="latin1")


def preprocess_checkins(df):
    processed = df.copy()
    processed["utc_time"] = pd.to_datetime(
        processed["utc_timestamp"],
        format="%a %b %d %H:%M:%S +0000 %Y",
    )
    processed["local_time"] = processed["utc_time"] + pd.to_timedelta(
        processed["timezone_offset"],
        unit="m",
    )
    processed["date"] = processed["local_time"].dt.date
    processed["hour"] = processed["local_time"].dt.hour
    processed["day_of_week"] = processed["local_time"].dt.day_name()
    return processed


def get_daily_counts(df, category_name):
    filtered = df[df["category_name"] == category_name]
    counts = filtered.groupby("date").size()
    return counts
