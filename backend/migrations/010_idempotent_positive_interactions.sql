DELETE FROM public.recipe_interactions older
USING public.recipe_interactions newer
WHERE older.user_id = newer.user_id
  AND older.recipe_id = newer.recipe_id
  AND older.interaction_type = newer.interaction_type
  AND older.interaction_type IN ('save', 'cook')
  AND (
      older.created_at < newer.created_at
      OR (older.created_at = newer.created_at AND older.id::text < newer.id::text)
  );

CREATE UNIQUE INDEX IF NOT EXISTS recipe_interactions_unique_positive_action_idx
    ON public.recipe_interactions (user_id, recipe_id, interaction_type)
    WHERE user_id IS NOT NULL AND interaction_type IN ('save', 'cook');
