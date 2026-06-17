ALTER TABLE public.user_taste_profiles
    ADD COLUMN IF NOT EXISTS centroid_vector double precision[],
    ADD COLUMN IF NOT EXISTS centroid_updated_at timestamp;

ALTER TABLE public.user_taste_profiles
    DROP CONSTRAINT IF EXISTS user_taste_profiles_centroid_dim;

ALTER TABLE public.user_taste_profiles
    ADD CONSTRAINT user_taste_profiles_centroid_dim
    CHECK (centroid_vector IS NULL OR array_length(centroid_vector, 1) = 25);

COMMENT ON COLUMN public.user_taste_profiles.centroid_vector IS
    'L2-normalized stabilized user centroid in the production PCA(25) cluster space.';

COMMENT ON COLUMN public.user_taste_profiles.centroid_updated_at IS
    'Timestamp watermark of the newest interaction incorporated into centroid_vector.';
