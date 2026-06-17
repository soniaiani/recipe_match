import { useCallback, useState } from 'react';
import { ApiError, forYouApi, TasteProfileResponse } from '@/services/api';

interface TasteProfileState {
  profile: TasteProfileResponse | null;
  loading: boolean;
  error: string | null;
}

export function useTasteProfile() {
  const [state, setState] = useState<TasteProfileState>({
    profile: null,
    loading: false,
    error: null,
  });

  const load = useCallback(async () => {
    setState(current => ({ ...current, loading: true, error: null }));
    try {
      const profile = await forYouApi.tasteProfile();
      setState({ profile, loading: false, error: null });
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Failed to load taste profile';
      setState({ profile: null, loading: false, error: msg });
    }
  }, []);

  return { ...state, load };
}
