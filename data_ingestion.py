import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.shares = [
            'NABIL', 'NRIC', 'SHIVM',  # Original shares
            'HBL', 'EBL', 'SCB', 'KBL', 'NMB', 'GBIME', 'NICA', 'PRVU', 'SBI', 'ADBL'  # 10 new shares
        ]
        self.stock_data = {}
        self.news_data = {}

    def fetch_stock_data(self, share):
        """
        Generate mock NEPSE stock data for the given share with unique trends.
        """
        try:
            logger.info(f"Generating mock NEPSE stock data for {share}")
            dates = pd.date_range(start="2025-02-02", end="2025-05-02", freq='D')
            num_days = len(dates)

            # Assign a unique base price for each share
            base_prices = {
                'NABIL': 500, 'NRIC': 450, 'SHIVM': 400,
                'HBL': 550, 'EBL': 480, 'SCB': 520, 'KBL': 390,
                'NMB': 510, 'GBIME': 470, 'NICA': 430, 'PRVU': 490,
                'SBI': 460, 'ADBL': 530
            }
            base_price = base_prices[share]

            # Create unique price trends using a combination of trends, sine waves, and noise
            np.random.seed(42 + hash(share) % 100)  # Unique seed for each share
            t = np.arange(num_days)

            # Base trend (linear or slight curve)
            trend = np.linspace(0, 50, num_days) if share in ['NABIL', 'HBL', 'NMB', 'SBI'] else \
                    np.linspace(-30, 30, num_days) if share in ['NRIC', 'EBL', 'GBIME', 'ADBL'] else \
                    np.linspace(-50, 0, num_days)

            # Add periodic fluctuations (sine waves with different frequencies and amplitudes)
            frequency = 0.1 + (hash(share) % 10) * 0.02  # Different frequency for each share
            amplitude = 20 + (hash(share) % 10) * 5       # Different amplitude for each share
            sine_wave = amplitude * np.sin(frequency * t)

            # Add random noise
            noise = np.random.normal(loc=0, scale=10, size=num_days)

            # Combine to create the closing price
            close_prices = base_price + trend + sine_wave + noise
            close_prices = np.clip(close_prices, base_price - 100, base_price + 100)

            stock_data = pd.DataFrame({
                'Date': dates,
                'Open': close_prices + np.random.uniform(-10, 10, len(dates)),
                'High': close_prices + np.random.uniform(0, 15, len(dates)),
                'Low': close_prices - np.random.uniform(0, 15, len(dates)),
                'Close': close_prices,
                'Volume': np.random.randint(80000, 150000, len(dates))
            })

            stock_data['High'] = stock_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
            stock_data['Low'] = stock_data[['Open', 'High', 'Low', 'Close']].min(axis=1)

            self.stock_data[share] = stock_data
            stock_data.to_csv(f"nepse_data_{share}.csv", index=False)
            logger.info(f"Saved NEPSE stock data for {share} to nepse_data_{share}.csv with {len(stock_data)} records")
        except Exception as e:
            logger.error(f"Error generating stock data for {share}: {e}")

    def fetch_news_data(self, share):
        """
        Generate mock news data for the given share.
        """
        try:
            logger.info(f"Generating mock news data for {share}")
            dates = pd.date_range(start="2025-02-02", end="2025-05-02", freq='D')
            titles = [
                f"{share} Rises After Positive Economic Report",
                f"{share} Announces Dividend Increase",
                f"Market Analysts Predict {share} Growth",
                f"Investors Cautious as {share} Volatility Increases",
                f"{share} Stock Surges on Strong Earnings",
                f"{share} Falls Amid Economic Uncertainty",
                f"{share} Faces Challenges with New Regulations",
                f"Market Analysts Warn of Potential {share} Decline"
            ]
            contents = [
                f"The Nepal Stock Exchange saw {share} rise today following a positive economic report.",
                f"{share} has announced a dividend increase, boosting investor confidence.",
                f"Market analysts are optimistic about {share}'s growth in the coming months.",
                f"Investors remain cautious as {share} experiences increased volatility.",
                f"{share} stock surged today after reporting strong quarterly earnings.",
                f"{share} dropped today due to concerns over economic uncertainty.",
                f"{share} is facing challenges due to new regulatory changes.",
                f"Analysts warn that {share} may decline if economic conditions worsen."
            ]

            news_data = []
            for i, date in enumerate(dates):
                idx = i % len(titles)
                news_data.append({
                    'date': date,
                    'title': titles[idx],
                    'content': contents[idx]
                })

            self.news_data[share] = pd.DataFrame(news_data)
            self.news_data[share].to_csv(f"news_data_{share}.csv", index=False)
            logger.info(f"Saved news data for {share} to news_data_{share}.csv with {len(self.news_data[share])} records")
        except Exception as e:
            logger.error(f"Error fetching news data for {share}: {e}")

    def run(self):
        """Run the data ingestion pipeline for all shares"""
        logger.info("Starting data ingestion pipeline")
        for share in self.shares:
            self.fetch_stock_data(share)
            self.fetch_news_data(share)
        logger.info("Data ingestion pipeline completed")

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()