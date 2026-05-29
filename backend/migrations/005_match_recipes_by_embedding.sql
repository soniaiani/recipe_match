CREATE OR REPLACE FUNCTION public.match_recipes_by_embedding(
  query_embedding vector(384),
  match_count int DEFAULT 100
)
RETURNS TABLE (
  id integer,
  similarity float
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    r.id,
    1 - (r.embedding <=> query_embedding) AS similarity
  FROM public.recipes r
  WHERE r.embedding IS NOT NULL
  ORDER BY r.embedding <=> query_embedding
  LIMIT match_count;
$$;
