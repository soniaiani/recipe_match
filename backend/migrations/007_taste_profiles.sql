CREATE TABLE IF NOT EXISTS public.recipe_cluster_vectors (
    recipe_id integer REFERENCES public.recipes(id) ON DELETE CASCADE,
    model_version text REFERENCES public.recipe_cluster_models(model_version) ON DELETE CASCADE,
    vector double precision[] NOT NULL,
    created_at timestamp DEFAULT now(),
    PRIMARY KEY (recipe_id, model_version),
    CONSTRAINT recipe_cluster_vectors_dim CHECK (array_length(vector, 1) = 25)
);

CREATE INDEX IF NOT EXISTS recipe_cluster_vectors_model_idx
    ON public.recipe_cluster_vectors (model_version);

CREATE TABLE IF NOT EXISTS public.recipe_cluster_profiles (
    model_version text REFERENCES public.recipe_cluster_models(model_version) ON DELETE CASCADE,
    cluster_id integer NOT NULL,
    centroid double precision[] NOT NULL,
    size integer NOT NULL DEFAULT 0,
    dominant_cuisine text,
    dominant_meal_type text,
    dominant_protein_type text,
    top_ingredients jsonb NOT NULL DEFAULT '[]',
    representative_recipes jsonb NOT NULL DEFAULT '[]',
    created_at timestamp DEFAULT now(),
    PRIMARY KEY (model_version, cluster_id),
    CONSTRAINT recipe_cluster_profiles_centroid_dim CHECK (array_length(centroid, 1) = 25)
);

CREATE TABLE IF NOT EXISTS public.user_taste_profiles (
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    model_version text REFERENCES public.recipe_cluster_models(model_version) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'ready',
    profile_signature text NOT NULL,
    interaction_count integer NOT NULL DEFAULT 0,
    positive_weight double precision NOT NULL DEFAULT 0,
    compact_summary text,
    description text,
    taste_cards jsonb NOT NULL DEFAULT '[]',
    top_clusters jsonb NOT NULL DEFAULT '[]',
    source text NOT NULL DEFAULT 'none',
    generated_at timestamp NOT NULL DEFAULT now(),
    expires_at timestamp NOT NULL,
    updated_at timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, model_version)
);

CREATE INDEX IF NOT EXISTS user_taste_profiles_expires_idx
    ON public.user_taste_profiles (expires_at);
