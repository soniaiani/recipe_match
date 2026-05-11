import { useState, useCallback } from 'react';
import { forYouApi, RecipeSummary, ApiError } from '@/services/api';

interface ForYouState {
  recipes: RecipeSummary[];
  loading: boolean;
  error: string | null;
}

export function useForYou() {
  const [state, setState] = useState<ForYouState>({
    recipes: [],
    loading: false,
    error: null,
  });

  const load = useCallback(async () => {
    setState({ recipes: [], loading: true, error: null });
    try {
      const res = await forYouApi.get();
      setState({ recipes: res.recipes, loading: false, error: null });
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to load recommendations';
      setState({ recipes: [], loading: false, error: msg });
    }
  }, []);

  return { ...state, load };
}
