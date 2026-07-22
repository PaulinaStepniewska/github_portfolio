# Aioli Menu Analysis Project

Comprehensive data analysis of Aioli restaurant menus across Poland, enriched with nutritional information and Nutri-Score calculations.

## Project Overview

This project analyzes menu items from Aioli restaurants in Poland by:
1. **Scraping** official restaurant websites
2. **Preprocessing** raw data (cleaning prices, weights, categories)
3. **Enriching** recipes with detailed ingredients using GPT-4
4. **Calculating** nutritional values per dish
5. **Scoring** items with the official Nutri-Score algorithm (ANSES 2022)
6. **Analyzing** relationships between price, nutrition, and healthiness

## Project Structure

```
aioli_project/
│
├── data/                        # Raw and processed data files
│   ├── dania.json              # Raw dishes data
│   ├── dishes.csv              # Cleaned dishes
│   └── ingredients.json        # Ingredient nutritional data
│
├── src/                         # Main source code modules
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration, constants, API keys
│   ├── scraper.py              # AioliScraper class for web scraping
│   ├── preprocessing.py        # DataPreprocessor for data cleaning
│   ├── ai_enrichment.py        # GPTPrompt & AIEnricher for recipe expansion
│   ├── nutriscore.py           # NutriScoreCalculator class
│   └── utils.py                # Utility functions (validation, logging, etc.)
│
├── eda/                         # Exploratory Data Analysis notebooks
│   ├── eda_overview.ipynb      # General data overview
│   └── eda_nutriscore.ipynb    # Nutri-Score analysis and correlations
│
├── results/                     # Output files from pipeline
│   └── final_analysis.csv      # Final results with Nutri-Score grades
│
├── main.py                      # Orchestration script - runs full pipeline
├── config.py                    # Global configuration
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file or set environment variable:
```bash
export API_KEY="your-openai-api-key"
```

### 3. Run Full Pipeline

```bash
python main.py
```

This will:
- Scrape all 4 locations (Warszawa Chmielna, Warszawa Świętokrzyska, Gdańsk, Katowice)
- Clean and preprocess data
- Expand recipes with GPT-4
- Get nutritional information
- Calculate Nutri-Score for all items
- Generate analysis reports

## Module Documentation

### `scraper.py` - AioliScraper
Handles web scraping from Aioli restaurant websites.

**Main Class:** `AioliScraper`
- `extract_section()` - Extract menu items from specific section
- `scrape_all()` - Scrape all locations and sections

### `preprocessing.py` - DataPreprocessor
Cleans and preprocesses raw data.

**Main Class:** `DataPreprocessor`
- `clean_and_prepare()` - Full preprocessing pipeline
  - Removes null prices
  - Cleans price column (comma → dot)
  - Extracts weight and unit
  - Creates dish IDs
  - Removes problematic dishes (e.g., "ZUPA DNIA")
  - Selects relevant columns

### `ai_enrichment.py` - AI-Powered Enrichment
Uses OpenAI GPT to expand recipes and get nutritional data.

**Main Classes:**
- `GPTPrompt` - Manages API communication
  - `get_batch_recipes()` - Expands ingredient lists with weights
  - `get_batch_ingredients()` - Gets nutritional values per 100g
- `AIEnricher` - Orchestrates enrichment pipeline

### `nutriscore.py` - NutriScoreCalculator
Implements official Nutri-Score algorithm (ANSES 2022).

**Main Class:** `NutriScoreCalculator`
- `calc_negative_points()` - Points for unhealthy nutrients
- `calc_positive_points()` - Points for healthy nutrients
- `calc_final_score()` - Final Nutri-Score calculation
- `score_to_grade()` - Converts score to grade (A-E)
- `calculate_nutriscore()` - Full pipeline for all dishes

**Nutri-Score Grades:**
- **A** (≤0 pts) - Excellent nutritional value
- **B** (1-2 pts) - Good
- **C** (3-10 pts) - Fair
- **D** (11-18 pts) - Poor
- **E** (>18 pts) - Very poor

### `utils.py` - Utilities
Helper functions for data processing and validation.

**Key Functions:**
- `chunk_dataframe()` - Split DataFrame into chunks
- `extract_json()` - Extract JSON from text
- `validate_ai_response()` - Check API response completeness
- `validate_nutrients()` - Validate nutrient values
- `save_dataframe()` / `load_dataframe()` - File I/O

## Data Pipeline

```
Raw Data → Scraper → Preprocessing → AI Enrichment → Nutrient Calc → Nutri-Score → Analysis
   ↓           ↓          ↓              ↓                ↓              ↓
 HTML      DataFrame   Cleaned        Expanded         Aggregated      Scored
         (191 items)  (valid)         Recipes          by Dish        (A-E)
```

## Key Metrics

- **Total Dishes:** 191 menu items
- **Unique Dishes:** ~48 unique names across locations
- **Ingredients:** 100+ unique ingredients
- **Locations:** 4 (Warszawa x2, Gdańsk, Katowice)
- **Price Range:** 22.99 - 145.99 PLN

## Analysis Features

The pipeline enables analysis of:

1. **Nutritional Quality**
   - Nutri-Score distribution by location
   - Correlation between price and nutrition
   - Healthiest vs unhealthiest items

2. **Menu Composition**
   - Category breakdown (main dishes, breakfast)
   - Ingredients
   - Fruits/vegetables percentage

3. **Health Insights**
   - Calorie distribution
   - Macro-nutrient balance
   - Sugar and salt content
   - Fiber content

## Development Notes

### Adding New Locations

Edit `config.py`:
```python
ALL_LOCATIONS = {
    "New_Location": "https://aioli.com.pl/menu-new-location/",
    # ... existing locations
}
```

### Customizing Menu Sections

Edit `config.py`:
```python
MENU_SECTIONS = ["Menu główne", "Śniadania"]
```

### Adjusting Nutri-Score Parameters

Edit `config.py` - modify `NUTRISCORE_CONFIG` thresholds.

### Using Individual Modules

```python
from src.scraper import AioliScraper
from src.preprocessing import DataPreprocessor
from src.ai_enrichment import AIEnricher
from src.nutriscore import NutriScoreCalculator

# Scrape
scraper = AioliScraper()
raw_df = scraper.scrape_all()

# Preprocess
preprocessor = DataPreprocessor()
clean_df = preprocessor.clean_and_prepare(raw_df)

# Enrich
enricher = AIEnricher()
recipes = enricher.enrich_recipes(clean_df)
```

## Logging

All operations produce detailed logs:

```
Scraping complete: 191 dishes from 4 locations
Preprocessing complete: Price cleaned, dish_id created
Recipe enrichment complete: 191 recipes expanded
Ingredient enrichment complete: 105 ingredients
Nutri-Score calculation complete: 191 dishes scored

Grade Distribution:
  A:   2 items (  1.0%) █
  B:   8 items (  4.2%) ██
  C:  45 items ( 23.6%) ███████████
  D:  98 items ( 51.3%) ██████████████████████████
  E:  38 items ( 19.9%) ██████████
```

## Security

- API keys stored in environment variables, not in code
- No credentials in version control
- Use `.env` file locally (add to `.gitignore`)

## Requirements

See `requirements.txt`:
- pandas
- requests
- beautifulsoup4
- openai
- seaborn
- matplotlib
- numpy

## License

This project is for portfolio purposes.

## Contributing

Suggestions welcome! Areas for enhancement:
- Additional restaurant chains
- Real-time menu monitoring
- Mobile app interface
- Meal recommendation engine
- Integration with other nutrition databases

---

**Created by:** Paulina Stępniewska  
**Last Updated:** May 2026
