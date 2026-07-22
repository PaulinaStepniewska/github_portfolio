"""
Configuration file for Aioli Project
Contains all constants, API keys, paths, and settings.
"""

import os
from pathlib import Path

# ===== DIRECTORIES =====
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
EDA_DIR = PROJECT_ROOT / "eda"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# ===== FILE PATHS =====
DISHES_CSV_PATH = DATA_DIR / "dishes.csv"
DISHES_JSON_PATH = DATA_DIR / "dishes.json"
INGREDIENTS_JSON_PATH = DATA_DIR / "ingredients.json"
INGREDIENTS_CHECK_PATH = DATA_DIR / "skladniki_dan_check.csv"

# ===== RESTAURANT LOCATIONS =====
ALL_LOCATIONS = {
    "Warszawa_Świętokrzyska": "https://aioli.com.pl/menu-warszawa-swietokrzyska/",
    "Warszawa_Chmielna": "https://aioli.com.pl/menu-warszawa-chmielna/",
    "Gdańsk": "https://aioli.com.pl/menu-gdansk/",
    "Katowice": "https://aioli.com.pl/menu-katowice/"
}

# Menu sections to scrape
MENU_SECTIONS = ["Menu główne", "Śniadania"]

# ===== API CONFIG =====
GPT_API_KEY = os.getenv("API_KEY")
GPT_MODEL = "gpt-4.1"
GPT_TEMPERATURE = 0.3
GPT_MAX_TOKENS = 8000
GPT_CHUNK_SIZE = 10  # Number of items per API request
GPT_REQUEST_DELAY = 2  # Seconds between requests

# ===== NUTRISCORE THRESHOLDS =====
# Source: ANSES 2022 https://www.santepubliquefrance.fr/nutri-score
NUTRISCORE_CONFIG = {
    "negative": {
        "energy_kj": {"step": 335, "max_pts": 10},
        "sugars": {"step": 3.4, "max_pts": 15},
        "saturated_fats": {"step": 1.0, "max_pts": 10},
        "salt": {"step": 0.2, "max_pts": 20},
    },
    "positive": {
        "protein": {"step": 2.4, "max_pts": 7},
        "fiber": {"step": 1.1, "max_pts": 5},
    },
    "fvl_points": [
        (80, 5),
        (60, 4),
        (40, 2),
        (0, 0),
    ],
    "n_threshold": 11,
}

# ===== LOGGING =====
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PIPELINE_LOG_JSON = LOGS_DIR / "pipeline_logs.json"
