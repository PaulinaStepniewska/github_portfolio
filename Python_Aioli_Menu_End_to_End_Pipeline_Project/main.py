"""
Main orchestration script for Aioli Project.
Coordinates scraping, preprocessing, AI enrichment, and analysis.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Import modules
from src.scraper import AioliScraper
from src.preprocessing import DataPreprocessor
from src.ai_enrichment import AIEnricher
from src.nutriscore import NutriScoreCalculator
from src.utils import (
    save_dataframe,
    load_dataframe,
    validate_ingredient_weights,
    validate_duplicate_dish_consistency,
    validate_recipe_ingredient_totals_from_df,
    configure_json_log_file,
)
from config import (
    DISHES_CSV_PATH,
    DISHES_JSON_PATH,
    INGREDIENTS_JSON_PATH,
    PIPELINE_LOG_JSON,
)


def prepare_dish_data(df):
    """Prepare dishes for ingredient expansion."""
    logger.info("\nPreparing dishes for enrichment...")
    
    # Select only necessary columns
    dishes = df[["dish_id", "dish_name", "dish_size", "unit", "price", "category", "location"]]
    save_dataframe(dishes, str(DISHES_CSV_PATH), format="csv")
    logger.info(f"Saved {len(dishes)} dishes to {DISHES_CSV_PATH}")
    
    return dishes


def process_ingredients(dish_ingredients_df):
    """Extract unique ingredients and create mapping."""
    logger.info("\n Processing ingredients...")
    
    # Explode ingredients and normalize
    df = dish_ingredients_df.explode(column="ingredients", ignore_index=True)
    df = df.join(pd.json_normalize(df["ingredients"]))
    df = df.drop(columns=["ingredients"])
    df.rename(columns={"name": "ingredient_name"}, inplace=True)
    
    # Create unique ingredients table
    ingredients = pd.DataFrame(
        df["ingredient_name"].unique(),
        columns=["ingredient_name"]
    )
    ingredients.insert(0, "ingredient_id", range(1, len(ingredients) + 1))
    
    logger.info(f"Created {len(ingredients)} unique ingredients")
    
    # Add ingredient_id to dish_ingredients
    df = df.merge(ingredients, on="ingredient_name", how="left")
    df = df.drop(columns=["ingredient_name"])
    
    return df, ingredients


def merge_nutritional_data(dish_ingredients_df, ingredients_df):
    """Merge dish-ingredient mappings with nutritional data."""
    logger.info("\nMerging nutritional data...")
    
    # Merge and calculate per-dish nutrients
    merged = dish_ingredients_df.merge(ingredients_df, how="left")
    
    # Calculate actual nutrients (from per-100g)
    nutrient_cols = [c for c in merged.columns if "_per_100g" in c]
    for col in nutrient_cols:
        new_col = col.replace("_per_100g", "")
        merged[new_col] = (merged[col] * merged["grams"] / 100).round(2)
    
    merged.drop(columns=nutrient_cols, inplace=True)
    
    # Calculate veg/fruit grams
    merged["grams_of_vegs_fruits"] = merged["grams"] * merged["is_fruit_veg"]
    
    logger.info(f"Merged {len(merged)} ingredient-dish records")
    return merged


def aggregate_dish_nutrients(merged_df, dishes_df):
    """Aggregate nutrients by dish."""
    logger.info("\nAggregating nutrients by dish...")
    
    # Sum nutrients by dish
    agg_dict = {
        col: "sum"
        for col in merged_df.columns
        if col not in ["dish_id", "ingredient_id", "ingredient_name", "grams"]
    }
    agg_dict["ingredient_id"] = "count"
    
    aggregated = merged_df.groupby("dish_id").agg(agg_dict).reset_index()
    aggregated.rename(columns={"ingredient_id": "ingredients_number"}, inplace=True)
    
    # Merge with dishes info
    aggregated = aggregated.merge(dishes_df, on="dish_id", how="left")
    
    # Calculate energy in kJ and veg/fruit percentage
    aggregated["energy (kJ)"] = (aggregated["calories"] / 4.184).round(2)
    aggregated["veg_fruit_percentage"] = (
        (aggregated["grams_of_vegs_fruits"] / aggregated["dish_size"]) * 100
    ).round(2)
    
    # Clean up
    aggregated = aggregated.drop(columns=["grams_of_vegs_fruits", "is_fruit_veg"])
    
    logger.info(f"Aggregated nutritional data for {len(aggregated)} dishes")
    return aggregated


def main():
    """Main orchestration function."""
    logger.info("\n" + "=" * 80)
    logger.info(" " * 20 + "AIOLI PROJECT - FULL PIPELINE")
    logger.info("=" * 80)
    
    try:
        configure_json_log_file(PIPELINE_LOG_JSON)

        # 1. SCRAPING
        logger.info("\n[1/5] SCRAPING MENUS")
        logger.info("-" * 80)
        scraper = AioliScraper()
        raw_df = scraper.scrape_all()
        logger.info(f"Raw data: {len(raw_df)} rows\n")
        
        # 2. PREPROCESSING
        logger.info("\n[2/5] PREPROCESSING DATA")
        logger.info("-" * 80)
        preprocessor = DataPreprocessor()
        clean_df = preprocessor.clean_and_prepare(raw_df)
        logger.info(f"After preprocessing: {len(clean_df)} unique dishes\n")
        
        # 3. RECIPE ENRICHMENT (GPT)
        logger.info("\n[3/5] ENRICHING RECIPES & INGREDIENTS (GPT)")
        logger.info("-" * 80)
        enricher = AIEnricher()
        
        # Prepare dishes
        dishes_df = prepare_dish_data(clean_df)
        logger.info(f"Dishes dataframe: {len(dishes_df)} rows\n")

        # Validate duplicate dish metadata consistency before enrichment
        if not validate_duplicate_dish_consistency(dishes_df, logger_obj=logger):
            logger.warning(
                "Znaleziono niespójne grupy dań o tym samym dish_name i dish_size. "
                "Sprawdź logi błędnych dopasowań przed dalszym przetwarzaniem."
            )
        
        # Expand recipes
        if 'ingredients' not in clean_df.columns:
            raise ValueError("Brak kolumny 'ingredients' w danych po preprocessing. Sprawdź pipeline wejściowy.")

        dish_ingredients_df = enricher.enrich_recipes(clean_df)
        logger.info(f"After enrich_recipes: {len(dish_ingredients_df)} rows\n")
        save_dataframe(dish_ingredients_df, str(DISHES_JSON_PATH), format="json")

        if dish_ingredients_df.empty:
            raise ValueError("Enrich_recipes zwróciło pusty DataFrame. Sprawdź odpowiedź GPT i popraw prompt lub dane wejściowe.")
        
        # Validate recipe ingredient totals before requesting ingredient nutrients from GPT
        is_valid = validate_recipe_ingredient_totals_from_df(
            dish_ingredients_df,
            dishes_df,
            logger_obj=logger,
            tolerance=0.0,
            stop_on_error=False,
        )
        if not is_valid:
            logger.warning(
                "Niektóre dania mają błędne sumy gramatur. Kontynuując...\n"
            )

        # Process ingredients
        dish_ingredients_mapped, unique_ingredients = process_ingredients(
            dish_ingredients_df
        )
        logger.info(f"After process_ingredients: {len(dish_ingredients_mapped)} ingredient-dish mappings\n")
        logger.info(f"Unique ingredients: {len(unique_ingredients)}\n")
        
        # Get nutritional data
        ingredients_with_nutrients = enricher.enrich_ingredients(unique_ingredients)
        save_dataframe(
            ingredients_with_nutrients,
            str(INGREDIENTS_JSON_PATH),
            format="json"
        )
        
        # 4. NUTRIENT CALCULATION
        logger.info("\n[4/5] CALCULATING NUTRIENTS")
        logger.info("-" * 80)
        merged = merge_nutritional_data(
            dish_ingredients_mapped, ingredients_with_nutrients
        )
        logger.info(f"After merge_nutritional_data: {len(merged)} rows\n")
        
        # Validate ingredient weights before aggregation
        merged = validate_ingredient_weights(merged, clean_df)
        
        nutrient_master = aggregate_dish_nutrients(merged, clean_df)
        logger.info(f"After aggregate_dish_nutrients: {len(nutrient_master)} dishes\n")
        
        # 5. NUTRI-SCORE CALCULATION
        logger.info("\n[5/5] CALCULATING NUTRI-SCORE")
        logger.info("-" * 80)
        calculator = NutriScoreCalculator()
        final_df = calculator.calculate_nutriscore(nutrient_master)
        
        # Save final results
        save_dataframe(final_df, "results/final_analysis.csv", format="csv")
        
        logger.info("\n" + "=" * 80)
        logger.info(" " * 25 + "PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nFinal Statistics:")
        logger.info(f"   • Total dishes analyzed: {len(final_df)}")
        logger.info(f"   • Unique ingredients: {len(unique_ingredients)}")
        logger.info(f"   • Locations: {final_df['location'].nunique()}")
        logger.info(f"   • Price range: {final_df['price'].min():.2f} - {final_df['price'].max():.2f} PLN")
        logger.info(f"\nOutput files:")
        logger.info(f"   • {DISHES_CSV_PATH}")
        logger.info(f"   • {DISHES_JSON_PATH}")
        logger.info(f"   • {INGREDIENTS_JSON_PATH}")
        logger.info(f"   • results/final_analysis.csv")
        logger.info("\n" + "=" * 80 + "\n")
        
        return final_df
        
    except Exception as e:
        logger.error(f"\nPIPELINE FAILED: {e}")
        logger.error("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    final_df = main()
