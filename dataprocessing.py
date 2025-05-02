import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import os
import tweepy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        logger.info("Twitter API not configured - using mock data")
        self.client = None

    def fetch_tweets(self, query, count=100, days_back=90):
        """Generate mock tweet data with varied sentiment"""
        logger.info("Generating mock tweet data for project")
        dates = pd.date_range(start="2025-02-02", end="2025-05-02", periods=days_back).to_list()
        
        # Extract share symbol from query for mock data
        share = query.split(' ')[0].replace('$', '')
        tweet_contents = [
            f"${share} is soaring after great earnings! #NEPSE",
            f"${share} might dip soon, be cautious. #NEPSE",
            f"${share} is stable today. #NEPSE",
            f"Excited about ${share}'s future growth! #NEPSE",
            f"Worried about ${share} with market uncertainty. #NEPSE"
        ]
        
        mock_tweets = pd.DataFrame({
            'date': [dates[i % len(dates)] for i in range(count)],
            'content': [tweet_contents[i % len(tweet_contents)] for i in range(count)],
            'user': [f"user_{i}" for i in range(count)],
            'retweets': [i % 15 for i in range(count)],
            'likes': [i % 30 for i in range(count)]
        })
        logger.info(f"Generated {len(mock_tweets)} mock tweets for {share}")
        return mock_tweets

class SentimentAnalyzer:
    def __init__(self):
        # Initialize VADER
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        logger.info("SentimentAnalyzer initialized with VADER")

    def analyze(self, text):
        """Analyze sentiment using VADER and return compound score"""
        if not isinstance(text, str):
            logger.warning(f"Invalid text for sentiment analysis: {text}")
            return 0.0
        try:
            scores = self.sia.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0

class DataProcessor:
    @staticmethod
    def process_text_data(text_df, analyzer, text_column='content'):
        """Process text data with sentiment analysis"""
        if text_df.empty or text_column not in text_df.columns:
            logger.warning("Empty or invalid text DataFrame received")
            return pd.DataFrame()

        try:
            text_df = text_df.copy()
            text_df['date'] = pd.to_datetime(text_df['date'])
            
            # Clean text
            text_df[text_column] = text_df[text_column].astype(str)
            text_df[text_column] = text_df[text_column].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
            text_df[text_column] = text_df[text_column].str.replace(r'\@\w+|\#\w+', '', regex=True)
            
            # Remove retweets if present
            if text_column == 'content':
                rt_mask = text_df[text_column].str.startswith('RT ')
                if rt_mask.any():
                    text_df = text_df[~rt_mask]
            
            # Perform sentiment analysis using compound score
            text_df['sentiment_score'] = text_df[text_column].apply(analyzer.analyze)
            logger.info(f"Processed {len(text_df)} text records with sentiment analysis")
            return text_df
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_financial_data(share):
        """Fetch NEPSE financial data for the given share"""
        try:
            data = pd.read_csv(f"nepse_data_{share}.csv")
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            logger.info(f"Fetched {len(data)} records of NEPSE data for {share}")
            return data
        except Exception as e:
            logger.error(f"Financial data fetch error for {share}: {e}")
            raise

    @staticmethod
    def calculate_metrics(financial_df):
        """Calculate financial metrics"""
        try:
            df = financial_df.copy()
            df['Daily_Return'] = df['Close'].pct_change()
            df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['Volatility'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
            df['P_E_Ratio'] = df['Close'] * 0.1
            logger.info("Calculated financial metrics")
            return df
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return financial_df

    @staticmethod
    def combine_data(tweet_df, news_df, financial_df):
        """Combine tweet sentiment, news sentiment, and financial data"""
        try:
            # Prepare tweet sentiment
            tweet_sentiment = pd.DataFrame()
            if not tweet_df.empty:
                tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date
                tweet_sentiment = tweet_df.groupby('date')['sentiment_score'].mean().reset_index()
                tweet_sentiment['date'] = pd.to_datetime(tweet_sentiment['date'])

            # Prepare news sentiment
            news_sentiment = pd.DataFrame()
            if not news_df.empty:
                news_df['date'] = pd.to_datetime(news_df['date']).dt.date
                news_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
                news_sentiment['date'] = pd.to_datetime(news_sentiment['date'])

            # Combine tweet and news sentiment
            combined_sentiment = pd.DataFrame()
            if not tweet_sentiment.empty and not news_sentiment.empty:
                combined_sentiment = pd.merge(
                    tweet_sentiment.rename(columns={'sentiment_score': 'tweet_sentiment'}),
                    news_sentiment.rename(columns={'sentiment_score': 'news_sentiment'}),
                    on='date',
                    how='outer'
                )
                combined_sentiment['combined_sentiment'] = combined_sentiment[['tweet_sentiment', 'news_sentiment']].mean(axis=1)
            elif not tweet_sentiment.empty:
                combined_sentiment = tweet_sentiment.rename(columns={'sentiment_score': 'combined_sentiment'})
            elif not news_sentiment.empty:
                combined_sentiment = news_sentiment.rename(columns={'sentiment_score': 'combined_sentiment'})
            
            # Merge with financial data
            financial_df = financial_df.reset_index()
            financial_df['date'] = pd.to_datetime(financial_df['Date']).dt.date
            financial_df['date'] = pd.to_datetime(financial_df['date'])
            
            if not combined_sentiment.empty:
                combined_df = pd.merge(
                    financial_df,
                    combined_sentiment[['date', 'combined_sentiment']],
                    on='date',
                    how='left'
                )
                combined_df['combined_sentiment'] = combined_df['combined_sentiment'].fillna(0)
            else:
                combined_df = financial_df
                combined_df['combined_sentiment'] = 0

            # Add sentiment features
            combined_df['Sentiment_Lag1'] = combined_df['combined_sentiment'].shift(1)
            sentiment_diff = combined_df['combined_sentiment'] - combined_df['combined_sentiment'].shift(5)
            sentiment_base = combined_df['combined_sentiment'].shift(5)
            combined_df['Sentiment_Trend'] = (sentiment_diff / sentiment_base.replace(0, np.nan)).fillna(0)
            combined_df = combined_df.fillna(0)

            combined_df = combined_df.set_index('date').sort_index()
            logger.info(f"Combined data with {len(combined_df)} records")
            return combined_df
        except Exception as e:
            logger.error(f"Data combining error: {e}")
            return pd.DataFrame()

def main(share='NABIL'):
    logger.info(f"Starting sentiment analysis pipeline for {share}")
    
    twitter_client = TwitterClient()
    analyzer = SentimentAnalyzer()
    processor = DataProcessor()
    
    # Fetch tweets
    tweet_data = twitter_client.fetch_tweets(
        query=f"${share} OR #{share} OR {share} -filter:retweets",
        count=200,
        days_back=90
    )
    
    # Process tweets
    processed_tweets = processor.process_text_data(tweet_data, analyzer, text_column='content')
    if not processed_tweets.empty:
        print(f"\n=== Processed Tweets for {share} ===")
        print(processed_tweets[['date', 'content', 'sentiment_score']].head())

    # Load and process news data
    try:
        news_data = pd.read_csv(f"news_data_{share}.csv")
        logger.info(f"Loaded {len(news_data)} news articles for {share}")
    except Exception as e:
        logger.warning(f"Error loading news_data_{share}.csv: {e}")
        news_data = pd.DataFrame()

    processed_news = processor.process_text_data(news_data, analyzer, text_column='content')
    if not processed_news.empty:
        print(f"\n=== Processed News for {share} ===")
        print(processed_news[['date', 'title', 'content', 'sentiment_score']].head())

    # Fetch and process financial data
    financial_data = processor.fetch_financial_data(share)
    financial_data = processor.calculate_metrics(financial_data)
    
    # Combine datasets
    combined_data = processor.combine_data(processed_tweets, processed_news, financial_data)
    if combined_data.empty:
        logger.error(f"Data combination failed for {share}")
        return

    # Save combined data
    combined_data.to_csv(f"combined_data_{share}.csv")
    logger.info(f"Saved combined data for {share} to combined_data_{share}.csv")

    print(f"\n=== Combined Dataset for {share} ===")
    print(combined_data.head())
    
    # Plot
    if not combined_data.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(combined_data.index, combined_data['Close'], label=f'{share} Close Price', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price (NRP)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(combined_data.index, combined_data['combined_sentiment'], label='Combined Sentiment Score', color='orange')
        ax2.set_ylabel('Sentiment Score', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title(f'{share} Stock Price and Combined Sentiment Score Over Time')
        ax1.grid(True)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(f'stock_sentiment_plot_{share}.png')
        logger.info(f"Plot saved as stock_sentiment_plot_{share}.png")
    
    logger.info(f"Pipeline completed successfully for {share}")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 50)
    shares = ['NABIL', 'NRIC', 'SHIVM', 'HBL', 'EBL', 'SCB', 'KBL', 'NMB', 'GBIME', 'NICA', 'PRVU', 'SBI', 'ADBL']
    for share in shares:
        main(share)