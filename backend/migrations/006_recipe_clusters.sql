CREATE TABLE IF NOT EXISTS public.recipe_cluster_models (
    model_version text PRIMARY KEY,
    params jsonb NOT NULL DEFAULT '{}'::jsonb,
    metrics jsonb,
    description text,
    created_at timestamp DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.recipe_clusters (
    recipe_id integer REFERENCES public.recipes(id) ON DELETE CASCADE,
    model_version text REFERENCES public.recipe_cluster_models(model_version) ON DELETE CASCADE,
    cluster_id integer NOT NULL,
    created_at timestamp DEFAULT now(),
    PRIMARY KEY (recipe_id, model_version)
);

CREATE INDEX IF NOT EXISTS recipe_clusters_recipe_idx
    ON public.recipe_clusters (recipe_id);

CREATE INDEX IF NOT EXISTS recipe_clusters_model_cluster_idx
    ON public.recipe_clusters (model_version, cluster_id);
