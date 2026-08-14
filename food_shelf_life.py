"""
=============================================================
  Cold Chain Project — Shelf Life Reference Data
=============================================================
  IMPORTANT — read this before trusting the LSTM's output:

  The raw dataset has sensor readings and a 3-class spoilage
  label (Fresh/At Risk/Spoiled), but NO ground-truth "hours of
  shelf life remaining" column. Since the LSTM needs a
  continuous target to regress against, this module derives
  one using a documented heuristic:

      remaining_hours = FOOD_MAX_SHELF_LIFE_HOURS[food]
                         * (1 - storage_risk_score)
                         - (storage_days * 24)
                         clipped to >= 0

  This combines: how long the food *could* last under ideal
  conditions, degraded by how risky its current conditions are
  (storage_risk_score, already computed elsewhere in the
  pipeline), minus time already elapsed.

  This is a reasonable proxy for demo/prototype purposes, but
  it is NOT measured ground truth. If real shelf-life outcome
  data becomes available (e.g. actual spoilage timestamps from
  field reports), retrain against that instead.

  FOOD_MAX_SHELF_LIFE_HOURS values are typical refrigerated
  shelf-life estimates (USDA / FDA food safety guidance,
  rounded), not guarantees for any specific product.
=============================================================
"""

FOOD_MAX_SHELF_LIFE_HOURS = {
    "Apple":      720,   # 30 days
    "Beef":       120,   # 5 days
    "Bread":      168,   # 7 days
    "Cheese":     504,   # 21 days
    "Chicken":    48,    # 2 days
    "Eggs":       504,   # 21 days
    "Fish":       48,    # 2 days
    "Milk":       168,   # 7 days
    "Mushroom":   168,   # 7 days
    "Orange":     504,   # 21 days
    "Potato":     720,   # 30 days
    "Spinach":    120,   # 5 days
    "Strawberry": 72,    # 3 days
    "Tomato":     168,   # 7 days
    "Yogurt":     336,   # 14 days
}

DEFAULT_MAX_SHELF_LIFE_HOURS = 168  # fallback for unseen food types


def compute_remaining_shelf_life_hours(food_name, storage_days, storage_risk_score):
    """
    Derives the synthetic regression target described above.
    Works on scalars or pandas Series (vectorized-friendly).
    """
    if hasattr(food_name, "map"):
        # Vectorized path: food_name is a pandas Series
        max_hours = food_name.map(FOOD_MAX_SHELF_LIFE_HOURS).fillna(DEFAULT_MAX_SHELF_LIFE_HOURS)
    else:
        # Scalar path: food_name is a single string
        max_hours = FOOD_MAX_SHELF_LIFE_HOURS.get(food_name, DEFAULT_MAX_SHELF_LIFE_HOURS)

    remaining = max_hours * (1 - storage_risk_score) - (storage_days * 24)
    return remaining.clip(lower=0) if hasattr(remaining, "clip") else max(0, remaining)