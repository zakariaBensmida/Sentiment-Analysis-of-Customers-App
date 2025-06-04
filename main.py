"""Main script for sentiment analysis."""

import logging
from pathlib import Path
from src.data_loader import load_and_preprocess_data
from src.model import train_and_predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the sentiment analysis pipeline."""
    logger.info("Starting sentiment analysis")
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    try:
        df = load_and_preprocess_data(data_dir / "raw" / "Tweets.csv")
        predictions_df = train_and_predict(df)
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        logger.info("Predictions saved")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
