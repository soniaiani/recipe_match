CREATE TABLE IF NOT EXISTS ingredient_graph (
  ingredient_a   TEXT NOT NULL,
  ingredient_b   TEXT NOT NULL,
  ppmi_score     FLOAT NOT NULL,
  co_occurrence  INTEGER NOT NULL,
  PRIMARY KEY (ingredient_a, ingredient_b)
);

CREATE INDEX IF NOT EXISTS ingredient_graph_a_score_idx
  ON ingredient_graph (ingredient_a, ppmi_score DESC);

CREATE TABLE IF NOT EXISTS ingredient_stats (
  ingredient     TEXT PRIMARY KEY,
  recipe_count   INTEGER NOT NULL,
  total_recipes  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ingredient_stats_recipe_count_idx
  ON ingredient_stats (recipe_count DESC);
