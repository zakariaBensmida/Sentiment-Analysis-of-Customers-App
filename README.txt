markdown
# Sentiment Analysis of Customer Tweets

A Python project for analyzing sentiment in airline tweets using NLP, with a Streamlit dashboard.

## Features
- Text preprocessing with NLTK
- Sentiment classification with scikit-learn
- Interactive Streamlit dashboard with Plotly
- Linting with black and ruff
- Environment management with python-dotenv
- Optional AWS S3 integration via aws_utils

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download `Tweets.csv` from [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) and place in `data/raw/`.
3. Run analysis: `python main.py`
4. View dashboard: `streamlit run src/app.py`

## Notes
- Global Python used; virtual environments recommended for production.
- Sensitive data (e.g., `.env`, `Tweets.csv`) excluded via `.gitignore`.
- AWS placeholders in `.env.public` for optional S3 integration (not used by default).

## Project Structure
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── output/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── app.py
│   └── aws_utils.py
├── tests/
├── .env.example
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
