import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '@/constants/api';

export type Suggestion = {
  ingredient: string;
  score?: number;
  ppmi_score?: number;
  scoring_method?: 'statistical' | 'ml_ltr' | string;
};

export type ExplorerRecipe = {
  id: number;
  name: string;
  description: string | null;
  image_url: string | null;
  prep_time: string | null;
  cook_time: string | null;
  total_time: string | null;
  servings: number | null;
  meal_type: string | null;
  cuisine: string | null;
  protein_type: string | null;
  ingredients_clean_str: string | null;
  is_vegetarian: boolean | null;
  is_vegan: boolean | null;
  is_gluten_free: boolean | null;
  is_dairy_free: boolean | null;
  jaccard_score: number;
};

type ApiEnvelope<T> = {
  data: T;
  error: string | null;
};

export function createExplorerSessionId(): string {
  const hex = (length: number) => Array.from(
    { length },
    () => Math.floor(Math.random() * 16).toString(16),
  ).join('');

  return `${hex(8)}-${hex(4)}-4${hex(3)}-${(8 + Math.floor(Math.random() * 4)).toString(16)}${hex(3)}-${hex(12)}`;
}

async function explorerRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = await AsyncStorage.getItem('auth_token');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers, signal: controller.signal });
    const body = await res.json() as ApiEnvelope<T>;
    if (!res.ok || body.error) {
      throw new Error(body.error ?? `HTTP ${res.status}`);
    }
    return body.data;
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Request timed out');
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export const explorerApi = {
  search: (query: string) =>
    explorerRequest<{ ingredients: string[] }>(`/explorer/search?q=${encodeURIComponent(query)}`),

  start: (ingredient: string) =>
    explorerRequest<{ center: string; suggestions: Suggestion[]; recipe_count: number }>('/explorer/start', {
      method: 'POST',
      body: JSON.stringify({ ingredient }),
    }),

  expand: (selectedIngredients: string[], sessionId?: string | null) =>
    explorerRequest<{ suggestions: Suggestion[]; recipe_count: number; relaxed: boolean }>('/explorer/expand', {
      method: 'POST',
      body: JSON.stringify({
        selected_ingredients: selectedIngredients,
        session_id: sessionId,
      }),
    }),

  recommend: (selectedIngredients: string[], sessionId?: string | null) =>
    explorerRequest<{ recipes: ExplorerRecipe[]; recipe_count: number; relaxed: boolean }>('/explorer/recommend', {
      method: 'POST',
      body: JSON.stringify({
        selected_ingredients: selectedIngredients,
        session_id: sessionId,
        finalize: true,
      }),
    }),
};
