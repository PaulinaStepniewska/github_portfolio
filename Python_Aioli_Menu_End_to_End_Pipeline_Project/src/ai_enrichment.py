"""
AI enrichment module for expanding recipes and calculating nutritional values.
Handles API communication with OpenAI GPT.
"""

import json
import time
import logging
from typing import List, Dict
import pandas as pd
from openai import OpenAI
from src.utils import (
    chunk_dataframe,
    create_recipe_key,
    extract_json,
    validate_ai_response,
    validate_recipe_ingredient_totals,
)
from config import (
    GPT_API_KEY,
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS,
    GPT_CHUNK_SIZE,
    GPT_REQUEST_DELAY,
)


class GPTPrompt:
    """Manages GPT API communication for recipe and ingredient enrichment."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=GPT_API_KEY)
        self.model = GPT_MODEL
        self.temperature = GPT_TEMPERATURE
        self.max_tokens = GPT_MAX_TOKENS
        self.chunk_size = GPT_CHUNK_SIZE
        self.request_delay = GPT_REQUEST_DELAY

    def call_gpt(self, prompt: str) -> str:
        """
        Call OpenAI GPT API.

        Args:
            prompt: Prompt text

        Returns:
            API response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT API Error: {e}")
            raise

    def get_batch_recipes(self, df: pd.DataFrame) -> str:
        """
        Generate detailed recipes with expanded ingredients.

        Args:
            df: DataFrame with dish_id, ingredients, dish_size

        Returns:
            JSON string with expanded recipes
        """
        data = df[["dish_id", "dish_name", "ingredients", "dish_size"]].to_dict(orient="records")

        prompt = f"""Jesteś ścisłym generatorem JSON.

        Zwróć TYLKO prawidłowy JSON. Brak przeceny.Żadnych wyjaśnień.
        Brak tekstu przed i po formacie JSON.

        KONTEKST: To są dania z restauracji AïOLI w Polsce. Menu zawiera burgery, tosty, sałatki, 
        makarony, śniadania, desery i pizze. Wiele dań to zestawy — np. burger podawany z frytkami,
        tost z surówką, śniadanie z pieczywem i dodatkami.

        ZASADY:
        1. Użyj "dish_name" aby zrozumieć CO wchodzi w skład dania — nie tylko bazuj na "ingredients".
        2. Jeśli dish_name sugeruje zestaw (np. zawiera "+", "z frytkami", "set", lub to burger/kanapka 
        w restauracji) — uwzględnij side dishes (frytki ~150-180g, surówka ~60-80g, sos ~30g).
        3. Rozwiń składniki na podstawowe elementy (np. "skrzydełka w panierce" → kurczak, bułka tarta, mąka, olej, sól).
        4. Suma gramatur MUSI być DOKŁADNIE równa dish_size.
        5. Przed zwróceniem JSON sprawdź każde danie: sum(grams) == dish_size. Jeśli nie — dodaj brakujące 
        składniki (woda, sos, oliwa, pieczywo) aż suma się zgadza.
        6. Minimalna gramatura składnika: 1g.
        7. Dla pizzy: gramatura ciasta jest funkcją WYŁĄCZNIE średnicy — ta sama średnica = ta sama gramatura ciasta (bez wyjątków).

        Przykład z burgerem dish_size=310g:
        bułka 65g + wołowina 110g + ser 20g + sos 20g + warzywa 30g + frytki 150g... ← BŁĄD: 395g > 310g
        bułka 60g + wołowina 110g + ser 15g + sos 20g + warzywa 15g + sałata 10g + frytki 60g + sól 2g = 292g ← za mało
        bułka 65g + wołowina 120g + ser 20g + sos aioli 20g + sałata 15g + pomidor 15g + ogórek 10g + sól 5g = 270g... 
        KOREKTA: zwiększ główny składnik lub dodaj element aż = 310g

        {json.dumps(data, ensure_ascii=False)}

        FORMAT:
        [
        {{
            "dish_id": "...",
            "ingredients": [
            {{
                "name": "...",
                "grams": 0
            }}
            ]
        }}
        ]"""

        self.logger.info(f"Requesting recipe expansion for {len(df)} dishes...")
        response = self.call_gpt(prompt)
        self.logger.info(f"Recipe response received")
        return response

    def get_batch_ingredients(self, df: pd.DataFrame) -> str:
        """
        Get nutritional information for ingredients.

        Args:
            df: DataFrame with ingredient_id, ingredient_name

        Returns:
            JSON string with nutritional data
        """
        data = df[["ingredient_id", "ingredient_name"]].to_dict(orient="records")

        prompt = f"""Jesteś ścisłym generatorem JSON.

        Zwróć TYLKO prawidłowy JSON.
        Brak przeceny.
        Żadnych wyjaśnień.
        Brak tekstu przed i po formacie JSON.

        Dla każdego składnika podaj wartości na 100g:
        - calories
        - protein
        - fats
        - carbs
        - sugars (cukry proste)
        - saturated_fats (tłuszcze nasycone)
        - salt
        - fiber

        Dodatkowo:
        - is_fruit_veg = 1 jeśli produkt to owoc lub warzywo
        - is_fruit_veg = 0 jeśli nie

        Dla każdego składnika podaj wartości odżywcze na 100g.
        Używaj standardowych wartości z tabel USDA/NEVO — nie zgaduj.

        ZAKRESY REFERENCYJNE (dla kontroli):
        - olej roślinny: ~900 kcal, 0g białka, 100g tłuszczu, 0g węglowodanów
        - kurczak pierś surowa: ~110 kcal, 23g białka, 1g tłuszczu, 0g węglowodanów  
        - mąka pszenna: ~360 kcal, 10g białka, 1g tłuszczu, 75g węglowodanów
        - warzywa liściaste: ~20 kcal, 2g białka, 0g tłuszczu, 3g węglowodanów

        Jeśli wartość jest niepewna — podaj środek typowego zakresu, nie 0.

        {json.dumps(data, ensure_ascii=False)}

        FORMAT:
        [
        {{
            "ingredient_id": "...",
            "ingredient_name": "...",
            "calories_per_100g": 0,
            "protein_per_100g": 0,
            "fats_per_100g": 0,
            "carbs_per_100g": 0,
            "sugars_per_100g": 0,
            "saturated_fats_per_100g": 0,
            "salt_per_100g": 0,
            "fiber_per_100g": 0,
            "is_fruit_veg": 0
        }}
        ]"""

        self.logger.info(f"Requesting nutritional data for {len(df)} ingredients...")
        response = self.call_gpt(prompt)
        self.logger.info(f"Nutrition response received")
        return response



class AIEnricher:
    """Orchestrates recipe and ingredient enrichment using GPT."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpt = GPTPrompt()

    def enrich_recipes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand recipes with detailed ingredients and weights.
        Processes valid items immediately, retries invalid ones.

        Args:
            df: DataFrame with dishes

        Returns:
            DataFrame with expanded ingredients
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("RECIPE ENRICHMENT STARTED")
        self.logger.info("=" * 60)

        df = create_recipe_key(df)
        recipe_groups = df.groupby("recipe_key")["dish_id"].apply(list).to_dict()
        representative_df = df.drop_duplicates(subset=["recipe_key"]).copy()
        self.logger.info(
            f"Deduplicated recipes: {len(df)} -> {len(representative_df)} unique recipe keys"
        )

        chunks = chunk_dataframe(representative_df, self.gpt.chunk_size)
        all_results = []
        failed_dishes = []  # Track dishes that failed validation

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")

            try:
                response = self.gpt.get_batch_recipes(chunk)
                cleaned = extract_json(response)
                parsed = json.loads(cleaned)
                
                # Validate and split into valid/invalid
                validation_result = validate_recipe_ingredient_totals(
                    chunk,
                    parsed,
                    logger_obj=self.logger,
                    tolerance=0.0,
                    stop_on_error=False,
                )
                valid_items = validation_result["valid"]
                invalid_items = validation_result["invalid"]
                
                # Process valid items immediately
                for item in valid_items:
                    recipe_key = chunk.loc[
                        chunk["dish_id"] == item["dish_id"],
                        "recipe_key",
                    ].iloc[0]
                    for dish_id in recipe_groups.get(recipe_key, [item["dish_id"]]):
                        item_copy = item.copy()
                        item_copy["dish_id"] = dish_id
                        all_results.append(item_copy)
                
                # Track invalid items for retry
                if invalid_items:
                    self.logger.warning(
                        f"Chunk {i + 1}: {len(invalid_items)} dania nie przeszły walidacji. Dodaję do retry..."
                    )
                    failed_dishes.extend(invalid_items)
                
                missing = validate_ai_response(chunk, parsed)
                if missing:
                    self.logger.warning(f"Missing dish_id in chunk {i + 1}: {missing}")

            except Exception as e:
                self.logger.error(f"Error processing chunk {i + 1}: {e}")

            time.sleep(self.gpt.request_delay)
        
        # Retry failed dishes (single requests)
        if failed_dishes:
            self.logger.info(f"\nRetrying {len(failed_dishes)} failed dishes individually...")
            for failed_item in failed_dishes:
                dish_id = failed_item.get("dish_id")
                self.logger.info(f"  Retrying dish_id={dish_id}...")
                
                # Create single-item dataframe for retry
                retry_chunk = representative_df[representative_df["dish_id"] == dish_id]
                if retry_chunk.empty:
                    self.logger.warning(f"  Dish {dish_id} not found in representative_df")
                    continue
                
                try:
                    response = self.gpt.get_batch_recipes(retry_chunk)
                    cleaned = extract_json(response)
                    parsed = json.loads(cleaned)
                    
                    if parsed and len(parsed) > 0:
                        retry_item = parsed[0]
                        recipe_key = retry_chunk.loc[
                            retry_chunk["dish_id"] == retry_item["dish_id"],
                            "recipe_key",
                        ].iloc[0]
                        for mapped_dish_id in recipe_groups.get(recipe_key, [retry_item["dish_id"]]):
                            item_copy = retry_item.copy()
                            item_copy["dish_id"] = mapped_dish_id
                            all_results.append(item_copy)
                        self.logger.info(f"  Dish {dish_id}: retry zaakceptowany")
                    else:
                        self.logger.warning(f"  Dish {dish_id}: GPT zwróciło pustą odpowiedź")
                except Exception as e:
                    self.logger.error(f"  Dish {dish_id}: retry nieudany - {e}")
                
                time.sleep(self.gpt.request_delay)

        result_df = pd.DataFrame(all_results)
        self.logger.info("=" * 60)
        self.logger.info(f"RECIPE ENRICHMENT COMPLETE: {len(result_df)} entries")
        self.logger.info(f"  (Failed & re-requested: {len(failed_dishes)})")
        self.logger.info("=" * 60 + "\n")
        
        return result_df

    def enrich_ingredients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get nutritional information for ingredients.

        Args:
            df: DataFrame with unique ingredients

        Returns:
            DataFrame with nutritional data
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INGREDIENT ENRICHMENT STARTED")
        self.logger.info("=" * 60)
        
        chunks = chunk_dataframe(df, self.gpt.chunk_size)
        all_results = []

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")

            try:
                response = self.gpt.get_batch_ingredients(chunk)
                cleaned = extract_json(response)
                parsed = json.loads(cleaned)
                missing = validate_ai_response(chunk, parsed, id_col="ingredient_id")

                if missing:
                    self.logger.warning(f"Missing ingredient_id in chunk {i + 1}: {missing}")

                all_results.extend(parsed)

            except Exception as e:
                self.logger.error(f"Error processing chunk {i + 1}: {e}")

            time.sleep(self.gpt.request_delay)

        result_df = pd.DataFrame(all_results)
        self.logger.info("=" * 60)
        self.logger.info(f"INGREDIENT ENRICHMENT COMPLETE: {len(result_df)} ingredients")
        self.logger.info("=" * 60 + "\n")
        
        return result_df

    def merge_nutritional_data(
        self, dish_ingredients: pd.DataFrame, ingredients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge dish-ingredient mappings with nutritional data.

        Args:
            dish_ingredients: Mapping of dishes to ingredients
            ingredients: Nutritional data per ingredient

        Returns:
            Merged DataFrame
        """
        merged = dish_ingredients.merge(ingredients, how="left")
        self.logger.info(f"Merged {len(merged)} ingredient records")
        return merged
