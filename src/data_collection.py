import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataCollector:
    def __init__(self, symbol=Config.SYMBOL):
        self.symbol = symbol
        self.raw_dir = Config.RAW_DATA_DIR

    def download_stock_data(self, start_date=Config.START_DATE, end_date=Config.END_DATE):
        """
        Download stock data from Yahoo Finance

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            pandas.DataFrame: Raw stock data
        """
        try:
            logging.info(f"Downloading {self.symbol} data from {start_date} to {end_date}")

            # Download data using yfinance
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,  # Keep raw prices and dividends separate
                prepost=False,  # No pre/post market data
            )

            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            # Clean column names
            df.columns = df.columns.str.replace(' ', '_')

            # Add some basic validation
            if len(df) < 100:
                logging.warning(f"Only {len(df)} days of data found. This might not be sufficient.")

            # Save raw data
            output_path = self.raw_dir / f"{self.symbol}_raw.csv"
            df.to_csv(output_path)

            logging.info(f"Successfully downloaded {len(df)} days of data")
            logging.info(f"Data saved to: {output_path}")
            logging.info(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            logging.error(f"Error downloading data: {str(e)}")
            raise

    def get_stock_info(self):
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info

            relevant_info = {
                'symbol': info.get('symbol'),
                'longName': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'marketCap': info.get('marketCap'),
                'currency': info.get('currency')
            }

            logging.info(f"Stock Info: {relevant_info}")
            return relevant_info

        except Exception as e:
            logging.warning(f"Could not fetch stock info: {str(e)}")
            return {}


def main():
    """Main function to run data collection"""
    collector = DataCollector()

    # Get stock info
    stock_info = collector.get_stock_info()

    # Download data
    df = collector.download_stock_data()

    print(f"\nData collection completed!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
