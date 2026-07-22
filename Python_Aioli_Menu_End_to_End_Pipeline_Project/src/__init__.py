"""
Aioli Project - src package
"""

from .scraper import AioliScraper
from .preprocessing import DataPreprocessor
from .ai_enrichment import AIEnricher, GPTPrompt
from .nutriscore import NutriScoreCalculator
from .utils import (
    chunk_dataframe,
    extract_json,
    validate_ai_response,
    validate_duplicate_dish_consistency,
    validate_nutrients,
    save_dataframe,
    load_dataframe,
    setup_logger,
)

__all__ = [
    "AioliScraper",
    "DataPreprocessor",
    "AIEnricher",
    "GPTPrompt",
    "NutriScoreCalculator",
    "chunk_dataframe",
    "extract_json",
    "validate_ai_response",
    "validate_duplicate_dish_consistency",
    "validate_nutrients",
    "save_dataframe",
    "load_dataframe",
    "setup_logger",
]
