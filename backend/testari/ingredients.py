from __future__ import annotations
import json
from typing import Any

DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"
OUTPUT_PATH = r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json"

INGREDIENT_FEATURE_MAP: dict[str, set[str]] = {
    "has_pasta": {
        "spaghetti", "egg noodle", "linguine", "fettuccine", "penne",
        "rotini", "elbow macaroni", "lasagna noodle", "pasta", "angel hair pasta",
        "rigatoni", "ziti", "orzo", "macaroni", "cheese tortellini", "gnocchi",
        "ditalini", "cavatappi pasta", "farfalle pasta", "shell pasta",
        "pasta shell", "pasta sauce", "cheese ravioli", "rice noodle",
        "ramen noodle", "rice vermicelli", "glass noodle", "chow mein noodle",
    },

    "has_rice": {
        "rice", "white rice", "brown rice", "arborio rice", "jasmine rice",
        "wild rice", "glutinous rice", "sushi rice", "yellow rice",
        "long grain white rice", "white rice",
    },

    "has_potato": {
        "potato", "sweet potato", "red potato", "hash brown",
        "potato flake", "hash brown potato", "tater tot", "mashed potato",
        "new potato", "yam",
    },

    "has_tomato_base": {
        "tomato", "tomato sauce", "tomato paste", "tomato juice",
        "cherry tomato", "salsa", "ketchup", "marinara sauce",
        "tomato soup", "crushed tomato", "diced tomato", "tomato puree",
        "stewed tomato", "grape tomato", "plum tomato", "pizza sauce",
        "enchilada sauce", "salsa verde", "tomatillo", "spaghetti sauce",
        "tomato vegetable juice", "tomato clam juice", "canned tomato",
        "sun dried tomato", "sun-dried tomato",
    },

    "has_cream_base": {
        "heavy cream", "sour cream", "cream cheese", "half and half",
        "evaporated milk", "condensed milk", "yogurt", "coconut milk",
        "whipped cream", "whipped topping", "cream", "greek yogurt",
        "mascarpone cheese", "ricotta", "ricotta cheese", "neufchatel cheese",
        "sweetened condensed milk", "coconut cream", "cream of tartar",
        "alfredo sauce", "lemon yogurt", "dulce de leche",
    },

    "has_cheese": {
        "parmesan cheese", "cheddar cheese", "mozzarella", "feta cheese",
        "swiss cheese", "monterey jack cheese", "cottage cheese",
        "cream cheese", "blue cheese", "goat cheese", "brie",
        "provolone", "gruyere cheese", "romano cheese", "asiago cheese",
        "pepper jack cheese", "colby cheese", "colby jack cheese",
        "american cheese", "mexican blend cheese", "italian cheese blend",
        "cheddar monterey jack cheese", "gorgonzola cheese", "gouda cheese",
        "pecorino romano cheese", "fontina", "mozzarella cheese",
        "processed cheese", "ricotta cheese", "queso fresco",
        "oaxaca cheese", "cotija cheese", "emmental cheese",
        "parmesan", "swiss chard",
    },

    "has_broth_base": {
        "chicken broth", "vegetable broth", "beef broth",
        "chicken stock", "chicken bouillon", "beef bouillon",
        "vegetable stock", "beef stock", "chicken soup base",
        "vegetable bouillon", "beef consomme", "beef base",
        "onion soup mix", "french onion soup", "dashi",
    },

    "has_mushroom": {
        "mushroom", "cream of mushroom soup", "portobello mushroom",
        "shiitake mushroom", "cremini mushroom", "oyster mushroom",
        "porcini mushroom", "morel mushroom", "chanterelle mushroom",
        "white mushroom", "golden mushroom soup",
    },

    "has_leafy_greens": {
        "spinach", "cabbage", "green bean", "broccoli", "kale",
        "lettuce", "romaine lettuce", "iceberg lettuce", "arugula",
        "bok choy", "swiss chard", "collard green", "chard",
        "mixed green", "salad green", "watercress", "red leaf lettuce",
        "napa cabbage", "green cabbage", "red cabbage", "radicchio",
        "endive", "butter lettuce",
    },

    "has_beans_legumes": {
        "black bean", "kidney bean", "chickpea", "pea",
        "lentil", "red lentil", "green lentil", "brown lentil",
        "pinto bean", "cannellini bean", "navy bean", "white bean",
        "great northern bean", "lima bean", "split pea", "black eyed pea",
        "refried bean", "baked bean", "edamame", "fava bean",
        "mung bean", "red kidney bean", "butter bean", "pork and beans",
        "chili bean",
    },

    "has_fruit": {
        "apple", "strawberry", "orange", "banana", "pineapple",
        "blueberry", "cranberry", "peach", "raspberry", "rhubarb",
        "mango", "lemon", "lime", "cherry", "grape", "pear",
        "watermelon", "pomegranate", "apricot", "plum", "fig",
        "kiwi", "nectarine", "blackberry", "date", "raisin",
        "dried cranberry", "dried apricot", "dried cherry",
        "maraschino cherry", "pineapple juice", "orange juice",
        "cherry pie filling", "blueberry pie filling", "peach pie filling",
        "mixed berry", "cantaloupe", "honeydew melon", "tangerine",
        "grapefruit", "persimmon", "quince",
    },

    "has_nuts": {
        "walnut", "pecan", "almond", "peanut butter", "peanut",
        "pine nut", "cashew", "hazelnut", "macadamia nut", "pistachio",
        "almond flour", "almond meal", "almond paste", "almond butter",
        "almond extract", "sunflower seed", "pumpkin seed", "sesame seed",
        "flaxseed", "chia seed", "poppy seed", "caraway seed",
        "flax meal", "coconut", "shredded coconut", "coconut flake",
    },

    "has_chocolate": {
        "cocoa powder", "semisweet chocolate chip", "chocolate chip",
        "chocolate", "bittersweet chocolate", "white chocolate",
        "milk chocolate", "white chocolate chip", "bittersweet chocolate chip",
        "chocolate bar", "cocoa nib", "chocolate hazelnut spread",
        "candy coated chocolate", "hot chocolate mix", "chocolate drink mix",
        "chocolate cake mix", "chocolate pudding mix", "chocolate sprinkle",
        "chocolate sandwich cookie",
    },

    "has_tortilla": {
        "flour tortilla", "corn tortilla", "tortilla", "tortilla chip",
        "corn tortilla chip", "taco shell", "tostada shell",
    },

    "has_spicy_ingredient": {
        "cayenne pepper", "chili powder", "jalapeno", "red pepper flake",
        "hot sauce", "green chile", "taco seasoning", "sriracha",
        "chili garlic sauce", "buffalo sauce", "cajun seasoning",
        "creole seasoning", "chipotle in adobo", "chipotle powder",
        "chipotle", "habanero pepper", "serrano pepper",
        "jalapeno pepper", "red chile", "red chile pepper",
        "ancho chile", "guajillo chile", "scotch bonnet pepper",
        "chili paste", "red curry paste", "gochujang", "gochugaru",
        "sambal oelek", "chili lime seasoning", "ancho chile powder",
        "green chile pepper", "chile pepper", "chili sauce",
        "chili oil", "thai chile", "chipotle chile in adobo",
        "jerk seasoning",
    },

    "has_asian_sauce": {
        "soy sauce", "sesame oil", "rice vinegar", "fish sauce",
        "oyster sauce", "hoisin sauce", "teriyaki sauce", "mirin",
        "rice wine", "sake", "tamari", "chili garlic sauce",
        "black bean sauce", "tamarind", "tamarind juice",
        "coconut milk", "lemongrass", "ginger", "five spice powder",
        "dashi", "miso", "bonito flake", "nori", "wakame",
        "gochujang", "sambal oelek",
    },
}


def normalize_ingredient(value: Any) -> str:
    return str(value).strip().lower()


def extract_ingredient_features(
    ingredients: list[str],
    name: str = "",
    description: str = "",
) -> dict[str, bool]:
    ingredients_set = {normalize_ingredient(i) for i in ingredients if i}
    text = f"{name} {description}".lower()

    features = {}
    for feature_name, keywords in INGREDIENT_FEATURE_MAP.items():
        normalized_keywords = {normalize_ingredient(k) for k in keywords}
        # Ingrediente — sursa primară
        from_ingredients = bool(ingredients_set & normalized_keywords)
        # Text (nume + descriere) — fallback
        from_text = any(kw in text for kw in INGREDIENT_FEATURE_MAP.get(feature_name, set()))
        features[feature_name] = from_ingredients or from_text

    return features


def main() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0

    for recipe in data:
        ingredients = recipe.get("ingredients_clean", [])

        if not isinstance(ingredients, list):
            ingredients = []

        new_features = extract_ingredient_features(
            ingredients,
            name=recipe.get("Name", ""),
            description=recipe.get("Description", ""),
        )

        if "llm_features" not in recipe or recipe["llm_features"] is None:
            recipe["llm_features"] = {}

        recipe["llm_features"].update(new_features)
        updated_count += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Dataset salvat în:")
    print(OUTPUT_PATH)
    print(f"Rețete procesate: {updated_count}")
    print(f"Features adăugate: {len(INGREDIENT_FEATURE_MAP)}")


if __name__ == "__main__":
    main()
# import json
# from collections import Counter

# DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"

# with open(DATASET_PATH, "r", encoding="utf-8") as f:
#     data = json.load(f)

# counter = Counter()

# for recipe in data:
#     ingredients = recipe.get("ingredients_clean", [])
#     if ingredients:
#         counter.update(ingredients)

# # Top 200 ingrediente
# top_ingredients = counter.most_common(1000)

# for ing, count in top_ingredients:
#     print(f"{ing}: {count}")