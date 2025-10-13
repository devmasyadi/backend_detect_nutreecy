# app/utils/nutrition_db.py
NUTRITION_DB = {
    "ayam goreng":   {"portion": "1 potong (~100 g)", "calories": 260, "protein": 28,  "carbs": 8,   "fat": 15},
    "bakso":         {"portion": "5 butir (~150 g)",  "calories": 220, "protein": 18,  "carbs": 14,  "fat": 10},
    "bakwan":        {"portion": "1 pc (~60 g)",      "calories": 180, "protein": 3,   "carbs": 15,  "fat": 12},
    "bubur ayam":    {"portion": "1 mangkuk",         "calories": 320, "protein": 14,  "carbs": 45,  "fat": 9},
    "nasi":          {"portion": "1 porsi (150 g)",   "calories": 204, "protein": 4,   "carbs": 45,  "fat": 0.4},
    "sambal":        {"portion": "1 sdm (15 g)",      "calories": 15,  "protein": 0.3, "carbs": 3,   "fat": 0.2},
    "sate":          {"portion": "5 tusuk (~150 g)",  "calories": 300, "protein": 25,  "carbs": 10,  "fat": 15},
    "tahu goreng":   {"portion": "1 potong (80 g)",   "calories": 140, "protein": 10,  "carbs": 6,   "fat": 8},
    "telur rebus":   {"portion": "1 butir (50 g)",    "calories": 78,  "protein": 6.3, "carbs": 0.6, "fat": 5.3},
    "tempe goreng":  {"portion": "1 potong (50 g)",   "calories": 180, "protein": 10,  "carbs": 8,   "fat": 10},
    "jeruk":         {"portion": "1 buah sedang (~130 g)", "calories": 62,  "protein": 1.2, "carbs": 15.4, "fat": 0.2},
    "apel":          {"portion": "1 buah sedang (~182 g)", "calories": 95,  "protein": 0.5, "carbs": 25,   "fat": 0.3},
    "alpukat":       {"portion": "1 buah sedang (~150 g)", "calories": 240, "protein": 3,   "carbs": 12.8, "fat": 22},
    "ketoprak":       {"portion": "1 buah sedang (~150 g)", "calories": 240, "protein": 3,   "carbs": 12.8, "fat": 22},
}

def get_nutrition(cls_name: str):
    key = cls_name.strip().lower()
    return NUTRITION_DB.get(key)
