
"""Train and predict sentiment model."""

import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Train logistic regression model and predict sentiments.

    Args:
        df: Preprocessed DataFrame with clean_text and airline_sentiment

    Returns:
        DataFrame with predictions
    """
    try:
        X = df["clean_text"]
        y = df["airline_sentiment"]
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_vec, y, df.index, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        result_df = pd.DataFrame({
            "text": df.loc[test_idx, "clean_text"].values,
            "true_sentiment": y_test.values,
            "predicted_sentiment": predictions
        })
        return result_df
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
