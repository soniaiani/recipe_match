import { useCallback, useRef, useState } from 'react';
import {
  ApiError, forYouApi, TasteProfileRecipesResponse,
} from '@/services/api';

interface TasteProfileRecipesState {
  data: TasteProfileRecipesResponse | null;
  loading: boolean;
  error: string | null;
}

export function useTasteProfileRecipes() {
  const lastRequestKey = useRef<string | null>(null);
  const [state, setState] = useState<TasteProfileRecipesState>({
    data: null,
    loading: false,
    error: null,
  });

  const load = useCallback(async (excludeRecipeIds: number[]) => {
    const requestKey = [...excludeRecipeIds].sort((a, b) => a - b).join(',');
    if (lastRequestKey.current === requestKey) return;
    lastRequestKey.current = requestKey;
    setState({ data: null, loading: true, error: null });
    try {
      const data = await forYouApi.tasteProfileRecipes(excludeRecipeIds);
      setState({ data, loading: false, error: null });
    } catch (err) {
      lastRequestKey.current = null;
      const message = err instanceof ApiError ? err.message : 'Failed to load taste recommendations';
      setState({ data: null, loading: false, error: message });
    }
  }, []);

  return { ...state, load };
}
