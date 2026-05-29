CREATE TABLE IF NOT EXISTS public.explorer_sessions (
    id uuid PRIMARY KEY,
    user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
    chain text[] NOT NULL DEFAULT '{}',
    started_at timestamp DEFAULT now(),
    updated_at timestamp DEFAULT now(),
    finalized_at timestamp
);

CREATE INDEX IF NOT EXISTS explorer_sessions_user_idx
    ON public.explorer_sessions (user_id, updated_at DESC);
