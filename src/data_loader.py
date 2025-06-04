"""Load and preprocess tweet data."""

import logging
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download("punkt")
nltk.download("stopwords")
logger = logging.getLogger(__name__)


def load_and_preprocess_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess tweet data.

    Args:
        file_path: Path to Tweets.csv

    Returns:
        Preprocessed DataFrame
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        stop_words = set(stopwords.words("english"))

        def clean_text(text):
            text = re.sub(
                r"http\S+|@\S+|#\S+", "", text
            )  # Remove URLs, mentions, hashtags
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
            return " ".join(tokens)

        df["clean_text"] = df["text"].apply(clean_text)
        df = df[["clean_text", "airline_sentiment"]].dropna()
        logger.info("Data preprocessed successfully")
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
