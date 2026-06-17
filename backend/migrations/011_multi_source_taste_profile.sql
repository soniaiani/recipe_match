ALTER TABLE public.user_taste_profiles
    ADD COLUMN IF NOT EXISTS behavior_centroid_vector double precision[],
    ADD COLUMN IF NOT EXISTS behavior_centroid_updated_at timestamp,
    ADD COLUMN IF NOT EXISTS source_weights jsonb NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS source_support jsonb NOT NULL DEFAULT '{}';

ALTER TABLE public.user_taste_profiles
    DROP CONSTRAINT IF EXISTS user_taste_profiles_behavior_centroid_dim;

ALTER TABLE public.user_taste_profiles
    ADD CONSTRAINT user_taste_profiles_behavior_centroid_dim
    CHECK (
        behavior_centroid_vector IS NULL
        OR array_length(behavior_centroid_vector, 1) = 25
    );

COMMENT ON COLUMN public.user_taste_profiles.behavior_centroid_vector IS
    'Stabilized centroid derived only from observed recipe interactions.';

COMMENT ON COLUMN public.user_taste_profiles.behavior_centroid_updated_at IS
    'Newest interaction incorporated into the behavioral centroid.';

COMMENT ON COLUMN public.user_taste_profiles.source_weights IS
    'Normalized Behavior, Find and Explorer weights used for the final centroid.';

COMMENT ON COLUMN public.user_taste_profiles.source_support IS
    'Evidence counts used to determine the multi-source Taste Profile weights.';
