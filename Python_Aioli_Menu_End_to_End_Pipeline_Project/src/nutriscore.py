"""
Nutri-Score calculation engine.
Implements the ANSES 2022 Nutri-Score algorithm.

Reference: https://www.santepubliquefrance.fr/nutri-score
"""

import pandas as pd
import logging
from config import NUTRISCORE_CONFIG

logger = logging.getLogger(__name__)


class NutriScoreCalculator:
    """Calculates Nutri-Score grades and points for food items."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = NUTRISCORE_CONFIG

    def _safe_pts(self, value: float, step: float, max_pts: int) -> int:
        """
        Calculate Nutri-Score points safely: treats NaN as 0.

        Args:
            value: Nutrient value
            step: Step size for points calculation
            max_pts: Maximum points allowed

        Returns:
            Calculated points (capped at max_pts)
        """
        if pd.isna(value) or step == 0:
            return 0
        return min(max_pts, int(value / step))

    def calc_negative_points(self, row: pd.Series) -> int:
        """
        Calculate negative points (higher = worse).
        Sums points for: energy, sugars, saturated fats, salt.

        Args:
            row: DataFrame row with nutrient values

        Returns:
            Total negative points
        """
        total = 0
        for nutrient, cfg in self.config["negative"].items():
            # Special case for energy
            value = row["energy (kJ)"] if nutrient == "energy_kj" else row[nutrient]
            total += self._safe_pts(value, cfg["step"], cfg["max_pts"])
        return total

    def calc_fvl_points(self, fvl_pct: float) -> int:
        """
        Calculate fruit/vegetable/legume points.

        Args:
            fvl_pct: Percentage of fruits, vegetables, legumes

        Returns:
            FVL points
        """
        if pd.isna(fvl_pct):
            return 0
        for threshold, pts in self.config["fvl_points"]:
            if fvl_pct > threshold:
                return pts
        return 0

    def calc_positive_points(self, row: pd.Series) -> int:
        """
        Calculate positive points (higher = better).
        Sums points for: protein, fiber, fruits/vegetables.

        Args:
            row: DataFrame row with nutrient values

        Returns:
            Total positive points
        """
        total = 0
        for nutrient, cfg in self.config["positive"].items():
            total += self._safe_pts(row[nutrient], cfg["step"], cfg["max_pts"])
        total += self.calc_fvl_points(row["veg_fruit_percentage"])
        return total

    def calc_final_score(self, row: pd.Series) -> int:
        """
        Calculate final Nutri-Score points.
        Uses different formulas based on negative points threshold.

        Args:
            row: DataFrame row with nutrient values

        Returns:
            Final Nutri-Score points
        """
        n = self.calc_negative_points(row)
        p = self.calc_positive_points(row)

        if n < self.config["n_threshold"]:
            return n - p
        else:
            # Alternative formula for high-energy items
            fiber_pts = self._safe_pts(
                row["fiber"],
                self.config["positive"]["fiber"]["step"],
                self.config["positive"]["fiber"]["max_pts"],
            )
            fvl_pts = self.calc_fvl_points(row["veg_fruit_percentage"])
            return n - (fiber_pts + fvl_pts)

    def score_to_grade(self, score: int) -> str:
        """
        Convert Nutri-Score points to grade (A-E).

        Args:
            score: Nutri-Score points

        Returns:
            Grade: A (best), B, C, D, E (worst)
        """
        if score <= 0:
            return "A"
        elif score <= 2:
            return "B"
        elif score <= 10:
            return "C"
        elif score <= 18:
            return "D"
        else:
            return "E"

    def calculate_nutriscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Nutri-Score for all items in DataFrame.

        Args:
            df: DataFrame with aggregated nutritional data

        Returns:
            DataFrame with nutriscore_points and nutriscore_grade columns
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("NUTRI-SCORE CALCULATION STARTED")
        self.logger.info("=" * 60)

        df["nutriscore_points"] = df.apply(self.calc_final_score, axis=1)
        df["nutriscore_grade"] = df["nutriscore_points"].apply(
            self.score_to_grade
        )

        # Log distribution
        grade_counts = df["nutriscore_grade"].value_counts().sort_index()
        self.logger.info("\nGrade Distribution:")
        for grade in ["A", "B", "C", "D", "E"]:
            count = grade_counts.get(grade, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            bar = "█" * int(pct / 5)  # Visual bar (max 20 chars)
            self.logger.info(f"  {grade}: {count:3d} items ({pct:5.1f}%) {bar}")

        self.logger.info("=" * 60)
        self.logger.info(f"NUTRI-SCORE CALCULATION COMPLETE: {len(df)} dishes scored")
        self.logger.info("=" * 60 + "\n")

        return df
