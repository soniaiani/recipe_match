CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.recipes
    ADD COLUMN IF NOT EXISTS embedding vector(384);

CREATE INDEX IF NOT EXISTS recipes_embedding_ivfflat_idx
    ON public.recipes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE OR REPLACE FUNCTION public.match_similar_recipes(
    target_recipe_id integer,
    match_count integer DEFAULT 20
)
RETURNS TABLE (
    id integer,
    name text,
    description text,
    image_url text,
    meal_type text,
    cuisine text,
    total_minutes double precision,
    is_vegetarian boolean,
    is_vegan boolean,
    is_gluten_free boolean,
    is_dairy_free boolean,
    is_quick boolean,
    similarity double precision
)
LANGUAGE sql
STABLE
AS $$
    WITH target AS (
        SELECT embedding
        FROM public.recipes
        WHERE id = target_recipe_id
          AND embedding IS NOT NULL
    )
    SELECT
        r.id,
        r.name,
        r.description,
        r.image_url,
        r.meal_type,
        r.cuisine,
        r.total_minutes,
        r.is_vegetarian,
        r.is_vegan,
        r.is_gluten_free,
        r.is_dairy_free,
        r.is_quick,
        1 - (r.embedding <=> target.embedding) AS similarity
    FROM public.recipes r, target
    WHERE r.id <> target_recipe_id
      AND r.embedding IS NOT NULL
    ORDER BY r.embedding <=> target.embedding
    LIMIT match_count;
$$;
