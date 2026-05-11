-- ── Task 1: Add ingredient feature columns to recipes ─────────────────────────
ALTER TABLE public.recipes
    ADD COLUMN IF NOT EXISTS has_pasta            boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_rice             boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_potato           boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_tomato_base      boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_cream_base       boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_cheese           boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_broth_base       boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_mushroom         boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_leafy_greens     boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_beans_legumes    boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_fruit            boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_nuts             boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_chocolate        boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_tortilla         boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_spicy_ingredient boolean DEFAULT false,
    ADD COLUMN IF NOT EXISTS has_asian_sauce      boolean DEFAULT false;

-- ── Task 1: Recommendation sessions ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.recommendation_sessions (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid        REFERENCES auth.users(id),
    created_at       timestamp   DEFAULT now(),
    completed_at     timestamp,
    answers          jsonb       DEFAULT '{}',
    question_order   text[],
    top_recipe_ids   integer[],
    questions_asked  integer     DEFAULT 0,
    entropy_final    double precision
);

-- ── Task 1: Recipe interactions ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.recipe_interactions (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid        REFERENCES auth.users(id),
    recipe_id        integer     REFERENCES public.recipes(id),
    interaction_type text,       -- 'view', 'like', 'save', 'cook', 'skip'
    weight           double precision DEFAULT 1.0,
    created_at       timestamp   DEFAULT now()
);
