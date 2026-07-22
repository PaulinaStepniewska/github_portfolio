"""
Data preprocessing and cleaning functions.
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def remove_null_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with null prices.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with null prices removed
        """
        initial_count = len(df)
        df = df.dropna(subset=["price"])
        removed = initial_count - len(df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} rows with null prices")
        else:
            self.logger.info("No null prices found")
            
        return df

    def clean_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price column: convert comma to dot and to float.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned prices
        """
        df["price"] = (
            df["price"].str.replace(",", ".", regex=False).astype(float)
        )
        self.logger.info(f"Price column cleaned: {df['price'].min():.2f} PLN - {df['price'].max():.2f} PLN")
        return df

    def extract_dish_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract weight and unit from dish_size column.
        Handles both weights (grams) and sizes (cm for pizzas).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with weight and unit columns
        """
        df[["dish_size", "unit"]] = df["dish_size"].str.extract(r"(\d+\.?\d*)\s*([a-zA-Z]+)?")
        df["dish_size"] = pd.to_numeric(df["dish_size"])
        
        pizzas = len(df[df["unit"].isin(["cm"])])
        if pizzas > 0:
            self.logger.info(f"Found {pizzas} pizzas (measured in cm, not grams)")
        
        self.logger.info("Weight extraction complete")
        return df

    def create_dish_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unique dish_id column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with dish_id column
        """
        df.insert(0, "dish_id", range(1, len(df) + 1))
        self.logger.info(f"Created dish_id for {len(df)} dishes")
        return df


    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only necessary columns for analysis.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with selected columns
        """
        columns = ["dish_id", "dish_name", "dish_size", "unit", "price", "category", "location", "ingredients"]
        df = df[columns]
        self.logger.info(f"Selected columns: {', '.join(columns)}")
        return df

    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        Args:
            df: Raw DataFrame from scraper

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PREPROCESSING STARTED")
        self.logger.info("=" * 60)
        
        df = self.remove_null_prices(df)
        df = self.clean_price_column(df)
        df = self.extract_dish_weight(df)
        df = self.create_dish_id(df)
        df = self.select_columns(df)
        
        self.logger.info("=" * 60)
        self.logger.info(f"PREPROCESSING COMPLETE: {len(df)} dishes ready")
        self.logger.info(f"  Unique dishes: {df['dish_name'].nunique()}")
        self.logger.info(f"  Locations: {df['location'].nunique()}")
        self.logger.info("=" * 60 + "\n")
        
        return df
