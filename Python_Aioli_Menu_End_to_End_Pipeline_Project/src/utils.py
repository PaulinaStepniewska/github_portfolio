"""
Utility functions for data processing, validation, and logging.
"""

import re
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import logging
from typing import List, Dict, Set, Any

# Setup logging
logger = logging.getLogger(__name__)


def setup_logger(name: str) -> logging.Logger:
    """Create a logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def chunk_dataframe(df: pd.DataFrame, size: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks of specified size.
    
    Args:
        df: DataFrame to chunk
        size: Chunk size
        
    Returns:
        List of DataFrames
    """
    return [df[i : i + size] for i in range(0, len(df), size)]


def extract_json(text: str) -> str:
    """
    Extract JSON string from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        JSON string
        
    Raises:
        ValueError: If no JSON found
    """
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in response")


def create_recipe_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable recipe key for deduplication across locations.
    """
    df = df.copy()

    def normalize_text(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

    df["recipe_key"] = (
        normalize_text(df.get("dish_name", pd.Series(dtype=str)))
        + "|"
        + normalize_text(df.get("dish_size", pd.Series(dtype=str)))
        + "|"
        + normalize_text(df.get("category", pd.Series(dtype=str)))
    )
    return df


def validate_ai_response(
    sent_df: pd.DataFrame, parsed_results: List[Dict], id_col: str = "dish_id"
) -> Set[Any]:
    """
    Check if AI returned all requested IDs.
    
    Args:
        sent_df: DataFrame sent to AI
        parsed_results: List of parsed AI responses
        id_col: ID column name
        
    Returns:
        Set of missing IDs
    """
    sent_ids = set(sent_df[id_col].tolist())
    returned_ids = set(item[id_col] for item in parsed_results)
    missing = sent_ids - returned_ids
    
    if missing:
        logger.warning(f"Missing {id_col} in AI response: {missing}")
    else:
        logger.info(f"AI Response Validation: All {len(sent_ids)} {id_col}s returned")
    
    return missing


def validate_recipe_ingredient_totals(
    sent_df: pd.DataFrame,
    parsed_results: List[Dict],
    id_col: str = "dish_id",
    size_col: str = "dish_size",
    ingredient_key: str = "ingredients",
    grams_key: str = "grams",
    tolerance: float = 0.0,
    logger_obj: logging.Logger = None,
    stop_on_error: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Validate that each GPT-expanded recipe has ingredient gram totals matching dish_size.
    Returns dict with 'valid' and 'invalid' keys.

    Args:
        sent_df: DataFrame originally sent to GPT
        parsed_results: Parsed GPT recipe responses
        id_col: ID column name
        size_col: Column containing target dish size
        ingredient_key: Key for ingredient list
        grams_key: Weight key inside ingredient dict
        tolerance: Allowed absolute difference
        logger_obj: Optional logger
        stop_on_error: If True, raises ValueError on any mismatch (deprecated)

    Returns:
        Dict with 'valid': [...], 'invalid': [...] lists
    """
    if logger_obj is None:
        logger_obj = logger

    dish_sizes = sent_df.set_index(id_col)[size_col].to_dict()
    valid_items = []
    invalid_items = []
    errors = []

    for item in parsed_results:
        dish_id = item.get(id_col)
        if dish_id not in dish_sizes:
            invalid_items.append(item)
            errors.append(f"Brak {id_col}={dish_id} w danych wejściowych")
            continue

        expected_size = float(dish_sizes[dish_id])
        ingredients = item.get(ingredient_key)
        if not isinstance(ingredients, list):
            invalid_items.append(item)
            errors.append(
                f"Dish {dish_id}: brak listy '{ingredient_key}' lub ma nieprawidłowy format"
            )
            continue

        total_grams = 0.0
        calc_error = False
        for ingredient in ingredients:
            try:
                total_grams += float(ingredient.get(grams_key, 0))
            except (TypeError, ValueError):
                invalid_items.append(item)
                errors.append(
                    f"Dish {dish_id}: nieprawidłowa gramatura w {ingredient}"
                )
                calc_error = True
                break
        
        if calc_error:
            continue

        if abs(total_grams - expected_size) > tolerance:
            invalid_items.append(item)
            diff = total_grams - expected_size
            errors.append(
                f"Dish {dish_id}: oczekiwano {expected_size}g, otrzymano {total_grams}g "
                f"(różnica {diff:+.2f}g)"
            )
        else:
            valid_items.append(item)

    if errors:
        logger_obj.warning(f"{len(errors)} błędów w walidacji gramatur chunku:")
        for error in errors[:10]:
            logger_obj.warning(f"  - {error}")
        if len(errors) > 10:
            logger_obj.warning(f"  ... i {len(errors) - 10} więcej")

    if stop_on_error and invalid_items:
        raise ValueError("GPT recipe totals validation failed")

    logger_obj.info(
        f"Chunk validation: {len(valid_items)} OK, {len(invalid_items)} błędnych"
    )
    return {"valid": valid_items, "invalid": invalid_items}


def validate_nutrients(df: pd.DataFrame, logger_obj: logging.Logger = None) -> bool:
    """
    Validate nutrient columns for NaNs, negative values, and outliers (>900).
    
    Args:
        df: DataFrame with nutrient data
        logger_obj: Logger instance
        
    Returns:
        True if validation passed, False otherwise
    """
    if logger_obj is None:
        logger_obj = logger
        
    nutrient_cols = [c for c in df.columns if "_per_100g" in c]
    issues_found = False
    
    for col in nutrient_cols:
        nulls = df[col].isna().sum()
        negatives = (df[col] < 0).sum()
        outliers = (df[col] > 900).sum()
        
        if any([nulls, negatives, outliers]):
            logger_obj.warning(
                f"{col}: {nulls} NaN, {negatives} negative, {outliers} outliers (>900)"
            )
            issues_found = True
    
    if not issues_found:
        logger_obj.info("Nutrient validation: All values OK")
        return True
    else:
        return False


def save_dataframe(df: pd.DataFrame, filepath: str, format: str = "csv") -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        format: File format ('csv' or 'json')
    """
    if format == "csv":
        df.to_csv(filepath, index=False)
        logger.info(f"Saved to CSV: {filepath}")
    elif format == "json":
        df.to_json(filepath)
        logger.info(f"Saved to JSON: {filepath}")
    else:
        raise ValueError(f"Unknown format: {format}")


def format_log_record_as_json(record: logging.LogRecord) -> str:
    """
    Convert a LogRecord into a JSON string.

    Args:
        record: LogRecord instance

    Returns:
        JSON string representation of the log record
    """
    payload = {
        "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
        "pathname": record.pathname,
        "funcName": record.funcName,
        "lineno": record.lineno,
    }

    if record.exc_info:
        formatter = logging.Formatter()
        payload["exception"] = formatter.formatException(record.exc_info)

    return json.dumps(payload, ensure_ascii=False)


class JsonLogFormatter(logging.Formatter):
    """Formatter that serializes log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        return format_log_record_as_json(record)


def configure_json_log_file(
    log_path: str | Path,
    logger_obj: logging.Logger = None,
    level: int = logging.INFO,
) -> logging.Handler:
    """
    Configure a JSON file handler for logging.

    Args:
        log_path: Path to the JSON log file
        logger_obj: Logger to attach the handler to (root logger if None)
        level: Logging level for the JSON handler

    Returns:
        Configured logging handler
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonLogFormatter())

    if logger_obj is None:
        logger_obj = logging.getLogger()

    logger_obj.addHandler(handler)
    return handler


def load_dataframe(filepath: str, format: str = "csv") -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filepath: Path to file
        format: File format ('csv' or 'json')
        
    Returns:
        Loaded DataFrame
    """
    if format == "csv":
        return pd.read_csv(filepath)
    elif format == "json":
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def validate_ingredient_weights(
    merged_df: pd.DataFrame,
    dishes_df: pd.DataFrame,
    tolerance: float = 0.05,
    rescale: bool = True,
    drop_grams: bool = True,
) -> pd.DataFrame:
    """
    Sprawdza czy suma gramatur poszczególnych składników się zgadza z wagą dania.
    Pomija pizzę (unit = 'cm').
    
    Jeśli suma jest niezgodna i rescale=True, przeskalowuje gramatury składników tak, aby suma zgadzała się z dish_size.
    
    Args:
        merged_df: DataFrame z ingredient-dish mappings (przed agregacją)
        dishes_df: DataFrame z informacją o dish_size i unit
        tolerance: Tolerancja przy dopasowaniu sumy gramatur
        rescale: Czy przeskalować gramatury dla dań niespełniających warunku
        drop_grams: Czy usunąć kolumnę 'grams' przed zwróceniem wyniku
        
    Returns:
        merged_df z ewentualnie przeskalowanymi gramaturami
    """
    logger.info("\nWalidacja gramatur składników...")
    
    # Merge aby mieć access do dish_size i unit
    validation_df = merged_df.merge(dishes_df[['dish_id', 'dish_size', 'unit']], on='dish_id', how='left')
    
    # Zsumuj gramy dla każdego dania
    grams_sum_per_dish = validation_df.groupby('dish_id')['grams'].sum().reset_index()
    grams_sum_per_dish.rename(columns={'grams': 'total_grams'}, inplace=True)
    
    # Merge z dish_size i unit
    grams_sum_per_dish = grams_sum_per_dish.merge(dishes_df[['dish_id', 'dish_size', 'unit']], on='dish_id', how='left')
    
    # Filtruj out pizzę (unit = 'cm')
    non_pizza = grams_sum_per_dish[grams_sum_per_dish['unit'] != 'cm'].copy()
    
    # Sprawdź czy suma się zgadza (z tolerancją ±5%)
    non_pizza['match'] = (
        (non_pizza['total_grams'] >= non_pizza['dish_size'] * (1 - tolerance)) &
        (non_pizza['total_grams'] <= non_pizza['dish_size'] * (1 + tolerance))
    )
    
    mismatches = non_pizza[non_pizza['match'] == False]
    
    if len(mismatches) > 0:
        logger.error(f"BŁĄD GRAMATUR: {len(mismatches)} dań ma niezgodną sumę składników!")
        logger.error("=" * 80)
        for _, row in mismatches.iterrows():
            diff = row['total_grams'] - row['dish_size']
            diff_pct = (diff / row['dish_size']) * 100 if row['dish_size'] > 0 else 0
            logger.error(
                f"   Dish ID {int(row['dish_id'])}: "
                f"oczekiwano {row['dish_size']}g, ale suma składników = {row['total_grams']}g "
                f"(różnica: {diff:+.1f}g, {diff_pct:+.1f}%)"
            )
        logger.error("=" * 80)

        if rescale:
            logger.info("Przeskalowywanie gramatur dla dań z błędną sumą...")
            for dish_id in mismatches['dish_id'].tolist():
                dish_size = float(dishes_df.loc[dishes_df['dish_id'] == dish_id, 'dish_size'].iloc[0])
                mask = merged_df['dish_id'] == dish_id
                total_grams = merged_df.loc[mask, 'grams'].sum()
                if total_grams <= 0:
                    logger.warning(f"Nie można przeskalować dish_id={dish_id}: suma gramów wynosi {total_grams}")
                    continue
                factor = dish_size / total_grams
                merged_df.loc[mask, 'grams'] = (merged_df.loc[mask, 'grams'] * factor).round(2)
            
            # Ponowna walidacja
            validation_df = merged_df.merge(dishes_df[['dish_id', 'dish_size', 'unit']], on='dish_id', how='left')
            grams_sum_per_dish = validation_df.groupby('dish_id')['grams'].sum().reset_index()
            grams_sum_per_dish.rename(columns={'grams': 'total_grams'}, inplace=True)
            grams_sum_per_dish = grams_sum_per_dish.merge(dishes_df[['dish_id', 'dish_size', 'unit']], on='dish_id', how='left')
            non_pizza = grams_sum_per_dish[grams_sum_per_dish['unit'] != 'cm'].copy()
            non_pizza['match'] = (
                (non_pizza['total_grams'] >= non_pizza['dish_size'] * (1 - tolerance)) &
                (non_pizza['total_grams'] <= non_pizza['dish_size'] * (1 + tolerance))
            )
            mismatches = non_pizza[non_pizza['match'] == False]
            if len(mismatches) == 0:
                logger.info("Przeskalowywanie zakończone sukcesem.")
            else:
                logger.error(f"Po przeskalowaniu nadal {len(mismatches)} dań ma błędną sumę gramatur.")
    else:
        logger.info(f"Gramatur walidacja: Wszystkie {len(non_pizza)} dania (bez pizzy) zgadzają się!")
    
    pizza_count = len(grams_sum_per_dish[grams_sum_per_dish['unit'] == 'cm'])
    if pizza_count > 0:
        logger.info(f"   (Pominięto {pizza_count} pizza/pizz - mają rozmiar zamiast wagi)")
    
    if drop_grams:
        if 'grams' in merged_df.columns:
            logger.info(f"\n Usuwanie kolumny 'grams' z merged DataFrame...")
            merged_df = merged_df.drop(columns=['grams'])
            logger.info(f"Kolumna 'grams' została usunięta")
    return merged_df


def validate_recipe_ingredient_totals_from_df(
    dish_ingredients_df: pd.DataFrame,
    dishes_df: pd.DataFrame,
    id_col: str = "dish_id",
    size_col: str = "dish_size",
    ingredient_col: str = "ingredients",
    grams_key: str = "grams",
    tolerance: float = 0.0,
    logger_obj: logging.Logger = None,
    stop_on_error: bool = False,
) -> bool:
    """
    Validate that each dish in a GPT-expanded recipe DataFrame has ingredient weights summing to dish_size.

    Args:
        dish_ingredients_df: DataFrame with columns dish_id and ingredients list
        dishes_df: DataFrame with dish_size and unit metadata
        id_col: Dish identifier column
        size_col: Dish size column
        ingredient_col: Column containing ingredient lists
        grams_key: Weight key inside each ingredient dict
        tolerance: Allowed absolute difference in grams
        logger_obj: Optional logger
        stop_on_error: If True, raises ValueError on mismatch

    Returns:
        True if all totals match, False otherwise
    """
    if logger_obj is None:
        logger_obj = logger

    required_cols = {id_col, ingredient_col}
    missing_cols = required_cols - set(dish_ingredients_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dish_ingredients_df: {missing_cols}")

    if size_col not in dishes_df.columns:
        raise ValueError(f"Missing column in dishes_df: {size_col}")
    if "unit" not in dishes_df.columns:
        raise ValueError("Missing column in dishes_df: unit")

    dish_meta = dishes_df.set_index(id_col)[[size_col, "unit"]].to_dict(orient="index")
    mismatches = []

    for _, row in dish_ingredients_df.iterrows():
        dish_id = row[id_col]
        if dish_id not in dish_meta:
            mismatches.append(f"Dish {dish_id}: brak danych metadanych w dishes_df")
            continue

        meta = dish_meta[dish_id]
        if meta["unit"] == "cm":
            continue

        ingredients = row[ingredient_col]
        if not isinstance(ingredients, list):
            mismatches.append(f"Dish {dish_id}: lista składników ma niepoprawny format")
            continue

        total_grams = 0.0
        for ingredient in ingredients:
            if not isinstance(ingredient, dict):
                mismatches.append(f"Dish {dish_id}: element składnika nie jest słownikiem: {ingredient}")
                total_grams = None
                break
            try:
                total_grams += float(ingredient.get(grams_key, 0))
            except (TypeError, ValueError):
                mismatches.append(
                    f"Dish {dish_id}: nieprawidłowa gramatura w składniku {ingredient}"
                )
                total_grams = None
                break

        if total_grams is None:
            continue

        expected_size = float(meta[size_col])
        if abs(total_grams - expected_size) > tolerance:
            mismatches.append(
                f"Dish {dish_id}: oczekiwano {expected_size}g, otrzymano {total_grams}g "
                f"(różnica {total_grams - expected_size:+.2f}g)"
            )

    if mismatches:
        logger_obj.warning(f"Walidacja składników: {len(mismatches)} dań z błędną sumą gramatur")
        for mismatch in mismatches[:10]:
            logger_obj.warning(f"  - {mismatch}")
        if len(mismatches) > 10:
            logger_obj.warning(f"  ... i {len(mismatches) - 10} więcej")
        if stop_on_error:
            raise ValueError("Recipe ingredient total validation failed before second GPT prompt")
        return False

    logger_obj.info("Recipe ingredient total validation: wszystkie dania mają poprawną sumę gramatur.")
    return True


def validate_duplicate_dish_consistency(
    dishes_df: pd.DataFrame,
    group_cols: List[str] = None,
    ignore_cols: List[str] = None,
    logger_obj: logging.Logger = None,
) -> bool:
    """
    Check that entries with the same dish_name and dish_size have identical values in other relevant columns.

    Args:
        dishes_df: DataFrame containing dish metadata
        group_cols: Columns used to identify the same recipe (default: ['dish_name', 'dish_size'])
        ignore_cols: Columns to ignore when comparing duplicates (default: ['dish_id', 'location'])
        logger_obj: Optional logger to use

    Returns:
        True if all duplicate groups are consistent, False otherwise
    """
    if logger_obj is None:
        logger_obj = logger

    if group_cols is None:
        group_cols = ["dish_name", "dish_size"]

    if ignore_cols is None:
        ignore_cols = ["dish_id", "location"]

    for col in group_cols:
        if col not in dishes_df.columns:
            raise ValueError(f"Missing required group column: {col}")

    compare_cols = [
        col
        for col in dishes_df.columns
        if col not in group_cols + ignore_cols
    ]

    logger_obj.info("\nSprawdzanie spójności duplikatów dla tych samych dish_name i dish_size...")
    inconsistent_groups = 0

    grouped = dishes_df.groupby(group_cols)
    for key, group in grouped:
        if len(group) <= 1:
            continue

        mismatch_columns = []
        for col in compare_cols:
            if group[col].nunique(dropna=False) > 1:
                values = group[col].astype(str).fillna("<NA>").unique().tolist()
                mismatch_columns.append((col, values))

        if mismatch_columns:
            inconsistent_groups += 1
            if isinstance(key, tuple):
                key_desc = ", ".join(f"{col}={val}" for col, val in zip(group_cols, key))
            else:
                key_desc = f"{group_cols[0]}={key}"

            logger_obj.error(f"Niespójna grupa dań: {key_desc}")
            logger_obj.error(f"  dish_id: {group['dish_id'].tolist()}")
            for col, values in mismatch_columns:
                logger_obj.error(f"  Kolumna '{col}' ma różne wartości: {values}")

    if inconsistent_groups == 0:
        logger_obj.info("Sprawdzanie duplikatów zakończone sukcesem. Brak niespójności.")
        return True

    logger_obj.error(f"Znaleziono {inconsistent_groups} niespójnych grup dań.")
    return False
