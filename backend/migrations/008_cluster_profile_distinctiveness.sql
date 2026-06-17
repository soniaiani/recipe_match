ALTER TABLE public.recipe_cluster_profiles
    ADD COLUMN IF NOT EXISTS top_ingredient_traits jsonb NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS top_boolean_traits jsonb NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS categorical_traits jsonb NOT NULL DEFAULT '{}';

COMMENT ON COLUMN public.recipe_cluster_profiles.top_ingredient_traits IS
    'Distinctive ingredient presence statistics ranked by prevalence * log2(lift + 1).';

COMMENT ON COLUMN public.recipe_cluster_profiles.top_boolean_traits IS
    'Distinctive culinary boolean statistics ranked by prevalence * log2(lift + 1).';

COMMENT ON COLUMN public.recipe_cluster_profiles.categorical_traits IS
    'Dominant cuisine, meal_type and protein_type with prevalence, global prevalence, lift and is_distinctive.';
