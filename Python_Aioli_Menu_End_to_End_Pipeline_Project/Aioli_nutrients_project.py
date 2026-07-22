# #  Aioli Menu Analysis
# 
# ## Project Goal
# The goal of this project is to analyze menu items from Aioli restaurants in Poland by enriching them with nutritional information such as:
# - Macro (protein, fats, carbs)
# - Caloric values
# - Nutri-Score 
# 
# 
# Based on this data, the project aims to explore relationships between:
# - Food healthiness and price
# - Differences in menu composition across cities in Poland
# - Overall nutritional quality of the restaurant's offerings

# Menu data is collected by scraping the official Aioli restaurant websites using `requests` and `BeautifulSoup`.
# 
# The dataset includes multiple locations in Poland:
# - Warszawa (Świętokrzyska, Chmielna)
# - Gdańsk
# - Katowice

# In[45]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import json
import os
import time
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt


# In[46]:


all_locations = {
    "Warszawa_Świętokrzyska": "https://aioli.com.pl/menu-warszawa-swietokrzyska/",
    "Warszawa_Chmielna": "https://aioli.com.pl/menu-warszawa-chmielna/",
    "Gdańsk": "https://aioli.com.pl/menu-gdansk/",
    "Katowice": "https://aioli.com.pl/menu-katowice/"}


# ### Data Extraction Logic
# A custom function is used to iterate through HTML elements and extract specific menu sections from each location webpage.
# 
# The analysis is limited to main dishes and breakfast items, as other sections contain either irrelevant information or inconsistent formats that would reduce data quality.
# 
# Each extracted dish is stored as a dictionary containing:
# - Category
# - Dish name
# - Portion size
# - Ingredients
# - Price
# - Location
# 
# 
# The results are aggregated into a single dataset

# In[47]:


def extract_section(section_name, location, url):

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    data = []

    for h2 in soup.find_all("h2"):
        if section_name in h2.get_text():
            section = h2.find_parent("div", class_="container")
            items = section.find_all("div", class_="py-20")

            for item in items:
                try:
                    h4 = item.find("h4")
                    if not h4:
                        continue

                    spans = h4.find_all("span")
                    dish_name = spans[0].get_text(strip=True) if len(spans) > 0 else None
                    dish_size = spans[1].get_text(strip=True) if len(spans) > 1 else None

                    p_tag = item.find("p")
                    ingredients = p_tag.get_text(strip=True) if p_tag else None

                    price_tag = item.find("span", class_="block")
                    price = price_tag.get_text(strip=True) if price_tag else None

                    data.append({
                        "category": section_name,
                        "dish_name": dish_name,
                        "dish_size": dish_size,
                        "ingredients": ingredients,
                        "price": price,
                        "location": location
                    })
                except Exception as e:
                    print(f"Error for {location}: {e}")
                    continue

    return data 


# In[48]:


all_data = []

for location, url in all_locations.items():
    try:
        all_data += extract_section("Menu główne", location, url)
        all_data += extract_section("Śniadania", location, url)
        print(f"Scraping complete for {location}")
    except Exception as e:
        print(f"Download failed for {location}: {e}")


# In[49]:


initial_df = pd.DataFrame(all_data)
initial_df.head()


# In[50]:


print(f'There are {initial_df.shape[0]} rows and {initial_df.shape[1]} columns in the dataset.')


# In[51]:


print(f'There are {initial_df['dish_name'].nunique()} unique dishes on the menu')


# In[52]:


initial_df.info()


# In[53]:


initial_df[initial_df['price'].isnull()]


# In[54]:


initial_df.dropna(subset=['price'], inplace=True)


# In[55]:


initial_df.info()


# Cleaning the price column to convert it into float:

# In[56]:


initial_df["price"] = (initial_df["price"].str.replace(",", ".", regex=False).astype(float))
initial_df.head(5)


# In[57]:


initial_df['price'].describe()


# In[58]:


print('Most expensive unique dishes on the menu:')
initial_df.groupby('dish_name')['price'].max().nlargest(5).reset_index()


# In[59]:


print('Cheapest dishes on the menu:')
initial_df.groupby('dish_name')['price'].max().nsmallest(5).reset_index()


# In[60]:


initial_df[initial_df['dish_name']=='ZUPA DNIA']


# In[61]:


initial_df = initial_df[initial_df['dish_name'] != 'ZUPA DNIA']


# ### Checking whether weights make sense:

# In[62]:


print(f'Weights of dishes on the menu: {initial_df['dish_size'].unique()}')
print(f'Minumum weight of a dish on the menu: {initial_df['dish_size'].unique().min()}')
print(f'Maximum weight of a dish on the menu: {initial_df['dish_size'].unique().max()}')


# In[63]:


pizzas_no = len(initial_df['dish_size'].where((initial_df['dish_size'] == '38 cm')|(initial_df['dish_size'] == '28 cm')).dropna())


# In[64]:


print(f'Pizza has size and not weight on the menu and there are {pizzas_no} such positions on the menu. We will handle this in the Chat GPT prompt.')


# In[65]:


initial_df[["dish_size", "unit"]] = initial_df["dish_size"].str.extract(r"(\d+\.?\d*)\s*([a-zA-Z]+)?")
initial_df["dish_size"] = pd.to_numeric(initial_df["dish_size"])


# In[66]:


initial_df.info()


# In[67]:


for col in initial_df.columns:
    print(col)
    print(initial_df[col].unique())
    print()


# In[68]:


initial_df.describe(include='all')


# ### Dataset Overview
# 
# The dataset consists of **191 menu items** across 4 restaurant locations, with no missing values.
# 
# ### Key characteristics:
# - 2 categories: main dishes and breakfast items
# - 48 unique dishes with some repetition across locations
# - Price ranges from **22.99 PLN to 145.99 PLN**, with an average of ~50 PLN
# 
# The data is well-structured and ready for further enrichment and exploratory analysis.

# ### Creating a unique identifier for dishes:

# In[69]:


initial_df.insert(0, "dish_id", [i for i in range(1, len(initial_df) + 1)])


# Check whether each menu item has the same number of items:

# In[70]:


initial_df['location'].value_counts()


# In[71]:


dishes = initial_df[['dish_id', 'dish_name', 'dish_size', 'unit', 'price','category', 'location']]
dishes.head()


# In[72]:


dishes.info()


# Save data to csv for manual review

# In[73]:


dishes.to_csv('data/dishes.csv')


# In[74]:


client = OpenAI(api_key=os.getenv("API_KEY"))

def get_batch_recipes(df):

    data = df[["dish_id", "ingredients", "dish_size"]].to_dict(orient="records")

    prompt = f"""Jesteś ścisłym generatorem JSON.

    Zwróć TYLKO prawidłowy JSON.
    Brak przeceny.
    Żadnych wyjaśnień.
    Brak tekstu przed i po formacie JSON.

    Każde danie musi zawierać szacunkową masę składników. Rozwiń listę składników, aby była bardziej szczegółowa, 
    tj. nie „skrzydełka z kurczaka w pikantnej panierce” tylko skrzydełko z kurczaka, bułka tarta, mąka, olej.
    Jeśli danie to pizza to oszacuj gramaturę na podstawie jej średnicy. Dla każdej pizzy o tej samej średnicy podaj taką samą gramaturę ciasta!

    Jeśli to ten sam przepis pojawiający się w różnych lokalizacjach (taka sama nazwa i ta sama wielkość), podaj dokładnie te same proporcje i masy składników.
    Nie twórz różnych wartości kalorycznych/makroskładników dla tej samej potrawy w różnych lokalizacjach.

    Najmniejsza możliwa gramatura składnika to 0.5g, a największa nie może przekraczać wagi dania.

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

    response = client.chat.completions.create(
    model="gpt-4.1",  
    messages=[{"role": "user", "content": prompt}],temperature=0.0,max_tokens=8000)

    return response.choices[0].message.content


# In[75]:


def get_batch_ingredients(df):

    data = df[["ingredient_id","ingredient_name"]].to_dict(orient="records")

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

    Jeśli nie jesteś pewien wartości → oszacuj realistycznie.

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


    response = client.chat.completions.create(
    model="gpt-4.1",  
    messages=[{"role": "user", "content": prompt}],temperature=0.0,max_tokens=8000)

    return response.choices[0].message.content


def create_recipe_key(df):
    df = df.copy()
    name = df["dish_name"].fillna("").astype(str)
    size = df["dish_size"].astype(str).fillna("")
    category = df["category"].fillna("").astype(str)

    def normalize_text(series):
        return series.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

    df["recipe_key"] = (
        normalize_text(name)
        + "|"
        + normalize_text(size)
        + "|"
        + normalize_text(category)
    )
    return df


# In[76]:


def chunk_dataframe(df, size=40):
    return [df[i:i+size] for i in range(0, len(df), size)]

def extract_json(text):
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in response")

def validate_ai_response(sent_df, parsed_results, id_col="dish_id"):
    '''Checks if the AI ​​returned a response for each sent dish_id. Returns a set of missing ids.'''
    sent_ids = set(sent_df[id_col].tolist())
    returned_ids = set(item[id_col] for item in parsed_results)
    missing = sent_ids - returned_ids
    if missing:
        print(f"Missing {id_col} in AI response: {missing}")
    else:
        print(f"AI Good Response Validation: All {len(sent_ids)} ids returned.")
    return missing

def validate_nutrients(df):
    '''Checks columns *per 100g for NaNs, negative values, and outliers (>900). Prints a summary of issues for each column.'''
    nutrient_cols = [c for c in df.columns if "_per_100g" in c]
    issues_found = False
    for col in nutrient_cols:
        nulls     = df[col].isna().sum()
        negatives = (df[col] < 0).sum()
        outliers  = (df[col] > 900).sum()
        if any([nulls, negatives, outliers]):
            print(f"{col}: {nulls} NaN, {negatives} negative, {outliers} outliers (>900)")
            issues_found = True
    if not issues_found:
        print("Component validation OK. No NaNs, negative values ​​or outliers.")


# In[77]:

small_df = initial_df.iloc[0:5]

df_with_keys = create_recipe_key(initial_df)
recipe_groups = df_with_keys.groupby("recipe_key")["dish_id"].apply(list).to_dict()
representative_df = df_with_keys.drop_duplicates(subset=["recipe_key"]).copy()

chunks = chunk_dataframe(representative_df, 20)
all_results = []

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}/{len(chunks)}")

    result = get_batch_recipes(chunk)

    if result:
        try:
            print("RESPONSE:\n", result)

            cleaned = extract_json(result)
            parsed = json.loads(cleaned)
            missing = validate_ai_response(chunk, parsed)
            if missing:
                print(f"Omitted dish_id in chunk {i+1}: {missing}")

            for item in parsed:
                rep_id = item.get("dish_id")
                recipe_key = representative_df.loc[representative_df["dish_id"] == rep_id, "recipe_key"].iloc[0]
                target_ids = recipe_groups.get(recipe_key, [rep_id])
                for target_id in target_ids:
                    expanded_item = item.copy()
                    expanded_item["dish_id"] = target_id
                    all_results.append(expanded_item)

        except Exception as e:
            print("JSON error:", e)

    time.sleep(2)

all_results


# In[78]:


dish_ingredients = pd.DataFrame(all_results)
dish_ingredients.head()


# In[79]:


dish_ingredients.to_json('data/dishes.json')


# In[80]:


dish_ingredients = dish_ingredients.explode(column = 'ingredients', ignore_index=True)
dish_ingredients = dish_ingredients.join(pd.json_normalize(dish_ingredients['ingredients']))
dish_ingredients = dish_ingredients.drop(columns=['ingredients'])
dish_ingredients.head()


# In[81]:


dish_ingredients_check = dish_ingredients.merge(dishes[['dish_id', 'dish_name']], on='dish_id', how='left')
dish_ingredients_check.to_csv('składniki_dań_check.csv')


# In[82]:


dish_ingredients.info()


# Checking whether the AI ​​has assigned reasonable weights to the ingredients:

# In[83]:


print(f'weights of ingredients in the dishes: {dish_ingredients['grams'].unique()}')
print(f'minumum weight of an ingredient: {dish_ingredients['grams'].unique().min()}')
print(f'maximum weight of an ingredient: {dish_ingredients['grams'].unique().max()}')


# In[84]:


dish_ingredients['name'][dish_ingredients['grams']<5].unique()


# In[85]:


dish_ingredients[dish_ingredients['grams']==0]


# In[86]:


dish_ingredients[dish_ingredients['grams']>=200]


# The fact that meat weight is high makes sense but tomatoes sound suspicious. Check what dish it is is required.

# In[87]:


dishes[dishes['dish_id'] == 13]


# It is a soup so it makes sense.

# In[88]:


dish_ingredients.rename(columns = {'name': 'ingredient_name'}, inplace=True)


# In[89]:


ingredients = pd.DataFrame(dish_ingredients['ingredient_name'].unique(),  columns=['ingredient_name'])
ingredients.insert(0, "ingredient_id",range(1, len(ingredients) + 1))
ingredients.head()


# In[90]:


dish_ingredients = dish_ingredients.merge(ingredients, on='ingredient_name', how='left')
dish_ingredients = dish_ingredients.drop(columns=['ingredient_name'])
dish_ingredients.head()


# In[91]:


chunks2 = chunk_dataframe(ingredients, 20)
all_results_ingredients = []

for i, chunk in enumerate(chunks2):
    print(f"Chunk {i+1}/{len(chunk)}")

    result = get_batch_ingredients(chunk)

    if result:
        try:
            print("RAW RESPONSE:\n", result) 

            cleaned = extract_json(result)
            parsed = json.loads(cleaned)
            missing = validate_ai_response(chunk, parsed, id_col="ingredient_id")
            if missing:
                print(f"Omitted ingredient_id in chunk{i+1}: {missing}")

            all_results_ingredients.extend(parsed)

        except Exception as e:
            print("JSON ERROR:", e)

    time.sleep(2)


# In[92]:


ingredients = pd.DataFrame(all_results_ingredients)
validate_nutrients(ingredients)
ingredients.head()


# In[93]:


ingredients.to_json('data/ingredients.json')


# In[94]:


ingredients.info()


# In[95]:


ingredients.describe()


# In[96]:


nutrient_cols = [col for col in ingredients.columns if '_per_100g' in col]
print(nutrient_cols)


# In[97]:


for col in nutrient_cols:
    max_value = ingredients[col].max()
    min_value = ingredients[col].min()

    max_ingredient = ingredients.loc[ingredients[col] == max_value,'ingredient_name'].to_list()
    min_ingredient = ingredients.loc[ingredients[col] == min_value,'ingredient_name'].to_list()

    print(f'\n{"="*60}')
    print(f'{col.upper()}')
    print(f'{"="*60}')

    print(f'Max value: {max_value}')
    print(f'Ingredients ({len(max_ingredient)}):')
    print(", ".join(max_ingredient[:10]))

    print()

    print(f'Min value: {min_value}')
    print(f'Ingredients ({len(min_ingredient)}):')
    print(", ".join(min_ingredient[:10]))


# Creating master DataFram with all required info to calculate Nutri-Score:

# In[98]:


nutrient_master_df = dish_ingredients.merge(ingredients, how = "left")
nutrient_master_df.head()


# In[99]:


for col in nutrient_cols:
    new_col = col.replace('_per_100g', '')  
    nutrient_master_df[new_col] = (nutrient_master_df[col] * nutrient_master_df['grams'] / 100).round(2)

nutrient_master_df.drop(columns = nutrient_cols, inplace = True)

nutrient_master_df.head()


# In[100]:


nutrient_master_df['grams_of_vegs_fruits'] = (nutrient_master_df['grams'] * nutrient_master_df['is_fruit_veg'])


# In[101]:


agg_dict = {col: 'sum' for col in nutrient_master_df.columns 
            if col not in ['dish_id', 'ingredient_id', 'ingredient_name']}

agg_dict['ingredient_id'] = 'count'

nutrient_master_df = nutrient_master_df.groupby('dish_id').agg(agg_dict).reset_index()
nutrient_master_df.rename(columns = {'ingredient_id' :'ingredients_number'}, inplace = True)
nutrient_master_df.head()


# In[102]:


nutrient_master_df = nutrient_master_df.merge(dishes, on = 'dish_id', how = 'left')
nutrient_master_df['energy (kJ)'] = round(nutrient_master_df['calories']/ 4.184,2)
nutrient_master_df['veg_fruit_percentage'] = (nutrient_master_df['grams_of_vegs_fruits'] / nutrient_master_df['dish_size']).round(2) * 100
nutrient_master_df = nutrient_master_df.drop(columns = ['grams_of_vegs_fruits', 'is_fruit_veg'])
nutrient_master_df.head()


# In[103]:


nutrient_master_df.info()


# In[104]:


# Source: NUTRISCORE THRESHOLDS (source :ANSES 2022  https://www.santepubliquefrance.fr/nutri-score)

nutriscore = {
    "negative": {
        "energy_kj": {"step": 335, "max_pts": 10},
        "sugars": {"step": 3.4, "max_pts": 15},
        "saturated_fats": {"step": 1.0, "max_pts": 10},
        "salt":  {"step": 0.2, "max_pts": 20},
    },
    "positive": {
        "protein": {"step": 2.4, "max_pts": 7},
        "fiber": {"step": 1.1, "max_pts": 5},
    },
    "fvl_points": [           
        (80, 5),
        (60, 4),
        (40, 2),
        (0,  0),
    ],
    "n_threshold": 11,}


# In[105]:


def _safe_pts(value, step, max_pts):
    '''Calculates Nutri-Score points safely: Treats NaN as 0.'''
    if pd.isna(value) or step == 0:
        return 0
    return min(max_pts, int(value / step))

def calc_negative_points(row):
    total = 0
    for nutrient, cfg in nutriscore["negative"].items():
        value = row[nutrient] if nutrient != "energy_kj" else row["energy (kJ)"]
        total += _safe_pts(value, cfg["step"], cfg["max_pts"])
    return total

def calc_fvl_points(fvl_pct):
    if pd.isna(fvl_pct):
        return 0
    for threshold, pts in nutriscore["fvl_points"]:
        if fvl_pct > threshold:
            return pts
    return 0

def calc_positive_points(row):
    total = 0
    for nutrient, cfg in nutriscore["positive"].items():
        total += _safe_pts(row[nutrient], cfg["step"], cfg["max_pts"])
    total += calc_fvl_points(row["veg_fruit_percentage"])
    return total

def calc_final_score(row):
    n = calc_negative_points(row)
    p = calc_positive_points(row)
    if n < nutriscore["n_threshold"]:
        return n - p
    else:
        fiber_pts = _safe_pts(row["fiber"], nutriscore["positive"]["fiber"]["step"],
                              nutriscore["positive"]["fiber"]["max_pts"])
        fvl_pts = calc_fvl_points(row["veg_fruit_percentage"])
        return n - (fiber_pts + fvl_pts)

def score_to_grade(score):
    if score <= 0:  return 'A'
    elif score <= 2:  return 'B'
    elif score <= 10: return 'C'
    elif score <= 18: return 'D'
    else:             return 'E'


# In[106]:


nutrient_master_df['nutriscore_points'] = nutrient_master_df.apply(calc_final_score, axis=1)
nutrient_master_df['nutriscore_grade'] = nutrient_master_df['nutriscore_points'].apply(score_to_grade)
nutrient_master_df.head()


# In[107]:


nutrient_master_df.describe()


# In[108]:


nutrient_master_df['nutriscore_grade'].value_counts().plot(kind='bar')


# Dishes with a poor nutri score definitely dominate in this menu.

# In[124]:


cities = nutrient_master_df['location'].unique()

fig, axes = plt.subplots(len(cities), 1, figsize=(8, 4 * len(cities)))
axes = np.atleast_1d(axes).flatten()

for ax, city in zip(axes, cities):

    nutrient_master_df.loc[
        nutrient_master_df['location'] == city,
        'nutriscore_grade'
    ].value_counts().sort_index().plot(
        kind='bar',
        ax=ax
    )

    ax.set_title(f'Nutri-Score distribution for {city}')
    ax.set_xlabel('Grade')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()


# In each location, dishes with a low nutri score dominate.

# In[110]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

nutrient_master_df.boxplot(column='nutriscore_points',by='category', ax=axes[0])
axes[0].set_title("Boxplot nutriscore_points by category")

stats = nutrient_master_df.groupby('category')['nutriscore_points'].describe()[['mean', 'std']]
stats.plot(kind='bar', ax=axes[1])

axes[1].set_title("Mean and STD by category")

plt.tight_layout()
plt.show()


# In[111]:


print('Top 5 healthiest dishes:')
nutrient_master_df.nsmallest(5, 'nutriscore_points')[['dish_name', 'nutriscore_grade', 'calories']]


# In[112]:


print('Top 5 unhealthiest dishes:')
nutrient_master_df.nlargest(5, 'nutriscore_points')[['dish_name', 'nutriscore_grade', 'calories']]


# In[113]:


corr = nutrient_master_df[['price','calories','protein','nutriscore_points']].corr()
print(corr)


# In[114]:


plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# In[115]:


sns.pairplot(nutrient_master_df[['price','calories','protein','nutriscore_points']])


# In[116]:


nutrient_master_df['location'].value_counts()


# Dataset is balanced, there is almost the same number of dishes across locations.

# In[117]:


price_comparison = nutrient_master_df.pivot_table(values='price', index='dish_name',columns='location', aggfunc='mean')
price_comparison.head()


# In[118]:


missing_dishes = price_comparison[price_comparison.isnull().any(axis=1)]
print(f"Number of dishes not available at all locations: {len(missing_dishes)}")
missing_dishes.head()


# In[119]:


inconsistent = price_comparison[price_comparison.nunique(axis=1) > 1]
print(f"Dishes with different prices across locations: {len(inconsistent)}")
inconsistent.head()


# In[120]:


print('Average dish prices per nutri-score grade:')
nutrient_master_df.groupby('nutriscore_grade')['price'].mean().round(2)

