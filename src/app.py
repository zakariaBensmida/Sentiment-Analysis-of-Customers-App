"""Streamlit app for sentiment analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")


def load_predictions() -> pd.DataFrame | None:
    """Load predictions."""
    predictions_path = Path("output/predictions.csv")
    if not predictions_path.exists():
        st.error("Run 'python main.py' to generate predictions.")
        logger.error(f"Missing: {predictions_path}")
        return None
    try:
        df = pd.read_csv(predictions_path)
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        logger.error(f"Load failed: {e}")
        return None


def main():
    """Run the Streamlit app."""
    st.title("Customer Sentiment Analysis Dashboard")
    st.markdown("Analyze sentiment of airline tweets.")

    predictions_df = load_predictions()
    if predictions_df is None:
        return

    st.header("Sentiment Distribution")
    fig = px.histogram(
        predictions_df,
        x="predicted_sentiment",
        title="Predicted Sentiment Counts",
        color="predicted_sentiment",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Sample Predictions")
    st.dataframe(predictions_df.head(10))


if __name__ == "__main__":
    main()
