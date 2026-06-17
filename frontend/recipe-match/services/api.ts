import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '@/constants/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DietaryProfile {
  is_vegetarian: boolean;
  is_vegan: boolean;
  is_gluten_free: boolean;
  is_dairy_free: boolean;
  excluded_ingredients: string[];
}

export interface UserProfile {
  id: string;
  email: string;
  dietary: DietaryProfile;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

export interface RecipeSummary {
  id: number;
  name: string;
  description: string | null;
  image_url: string | null;
  meal_type: string | null;
  cuisine: string | null;
  total_minutes: number | null;
  is_vegetarian: boolean;
  is_vegan: boolean;
  is_gluten_free: boolean;
  is_dairy_free: boolean;
  is_quick: boolean;
}

export interface RecipeDetail extends RecipeSummary {
  prep_time: string | null;
  cook_time: string | null;
  total_time: string | null;
  servings: number | null;
  ingredients: string | null;
  ingredients_clean: string[];
  directions: string | null;
  protein_type: string | null;
  is_nut_free: boolean;
  is_spicy: boolean;
  is_sweet: boolean;
  needs_oven: boolean;
  needs_stovetop: boolean;
  is_no_cook: boolean;
}

export interface Collection {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
}

export interface SavedRecipe {
  id: string;
  recipe_id: number;
  collection_id: string | null;
  saved_at: string;
  recipe: RecipeSummary | null;
}

export interface ShoppingListResponse {
  recipe_id: number;
  recipe_name: string;
  ingredients: string[];
}

export interface SimilarRecipe extends RecipeSummary {
  similarity: number;
}

export interface SimilarRecipesResponse {
  recipes: SimilarRecipe[];
}

export interface IngredientSuggestionsResponse {
  ingredients: string[];
}

export interface TasteCard {
  title: string;
  text: string;
  traits: string[];
  ingredients: string[];
  recipes: string[];
}

export interface TasteAttributeTrait {
  name: string;
  global_recipe_count: number;
  prevalence: number;
  global_prevalence: number;
  lift: number;
  score: number;
  is_globally_common: boolean;
}

export interface TasteCategoricalTrait {
  value: string;
  prevalence: number;
  global_prevalence: number;
  lift: number;
  is_distinctive: boolean;
}

export interface TasteCluster {
  cluster_id: number;
  weight: number;
  similarity: number;
  dominant_cuisine: string | null;
  dominant_meal_type: string | null;
  dominant_protein_type: string | null;
  top_ingredients: string[];
  top_ingredient_traits: TasteAttributeTrait[];
  top_boolean_traits: TasteAttributeTrait[];
  categorical_traits: Record<string, TasteCategoricalTrait>;
  representative_recipes: string[];
}

export interface TasteProfileResponse {
  status: 'ready' | 'insufficient_data' | 'unavailable';
  compact_summary: string | null;
  description: string | null;
  taste_cards: TasteCard[];
  top_clusters: TasteCluster[];
  generated_at: string | null;
  source: 'cache' | 'gemini' | 'fallback' | 'none';
}

export interface TasteClusterRecipes {
  cluster_id: number;
  recipes: RecipeSummary[];
}

export interface TasteProfileRecipesResponse {
  clusters: TasteClusterRecipes[];
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = await AsyncStorage.getItem('auth_token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const body = await res.json() as {
    data?: T;
    error?: string | null;
    detail?: string | { msg?: string }[];
  };

  if (!res.ok || body.error) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map(item => item.msg).filter(Boolean).join(', ')
      : body.detail;
    throw new ApiError(res.status, body.error ?? detail ?? `HTTP ${res.status}`);
  }
  return body.data as T;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export const authApi = {
  register: (email: string, password: string, dietary: DietaryProfile) =>
    request<AuthResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, dietary }),
    }),

  login: (email: string, password: string) =>
    request<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  logout: () =>
    request<null>('/auth/logout', { method: 'POST' }),

  me: () =>
    request<UserProfile>('/auth/me'),

  updateDietary: (dietary: DietaryProfile) =>
    request<UserProfile>('/auth/me', {
      method: 'PATCH',
      body: JSON.stringify(dietary),
    }),
};

export const recipesApi = {
  get: (id: number) =>
    request<RecipeDetail>(`/recipes/${id}`),

  shoppingList: (id: number) =>
    request<ShoppingListResponse>(`/recipes/${id}/shopping-list`),

  similar: (id: number, limit = 20) =>
    request<SimilarRecipesResponse>(`/recipes/${id}/similar?limit=${limit}`),

  ingredients: (q: string, limit = 20) =>
    request<IngredientSuggestionsResponse>(
      `/recipes/ingredients?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),
};

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

export const collectionsApi = {
  list: () =>
    request<Collection[]>('/collections'),

  create: (name: string) =>
    request<Collection>('/collections', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  delete: (id: string) =>
    request<null>(`/collections/${id}`, { method: 'DELETE' }),
};

// ---------------------------------------------------------------------------
// Saved recipes
// ---------------------------------------------------------------------------

export const savedApi = {
  save: (recipe_id: number, collection_id?: string) =>
    request<SavedRecipe>('/saved', {
      method: 'POST',
      body: JSON.stringify({ recipe_id, collection_id: collection_id ?? null }),
    }),

  unsave: (recipe_id: number) =>
    request<null>(`/saved/${recipe_id}`, { method: 'DELETE' }),

  list: () =>
    request<SavedRecipe[]>('/saved'),

  listByCollection: (collection_id: string) =>
    request<SavedRecipe[]>(`/saved/collections/${collection_id}`),
};

// ---------------------------------------------------------------------------
// For You
// ---------------------------------------------------------------------------

export const forYouApi = {
  get: () =>
    request<{ recipes: RecipeSummary[] }>('/foryou'),

  tasteProfile: () =>
    request<TasteProfileResponse>('/foryou/taste-profile'),

  tasteProfileRecipes: (excludeRecipeIds: number[] = []) => {
    const params = new URLSearchParams();
    excludeRecipeIds.forEach(id => params.append('exclude_recipe_ids', String(id)));
    const query = params.toString();
    return request<TasteProfileRecipesResponse>(
      `/foryou/taste-profile/recipes${query ? `?${query}` : ''}`,
    );
  },
};

// ---------------------------------------------------------------------------
// Recommendation session (Bayesian engine)
// ---------------------------------------------------------------------------

export interface RecProgress {
  current: number;
  max: number;
}

export interface RecQuestion {
  id: string;
  type: 'categorical' | 'multiselect' | 'boolean';
  options?: string[];
  any_option?: string;
}

export interface RecSessionStartResponse {
  session_id: string;
  question: RecQuestion;
  progress: RecProgress;
}

export interface RecScoredRecipe {
  id: number;
  name: string;
  image_url: string | null;
  meal_type: string | null;
  cuisine: string | null;
  protein_type: string | null;
  match_score: number;  // 0–100
}

export interface RecAnswerResponse {
  status: 'continue' | 'done';
  question: RecQuestion | null;
  entropy: number | null;
  questions_asked: number | null;
  progress: RecProgress | null;
  results: RecScoredRecipe[] | null;
  results_count: number | null;
}

export interface RecResultsResponse {
  results: RecScoredRecipe[];
  results_count: number;
}

export const recommendationApi = {
  start: (dietary?: DietaryProfile) =>
    request<RecSessionStartResponse>('/recommendations/session/start', {
      method: 'POST',
      body: JSON.stringify({
        dietary: dietary ?? null,
      }),
    }),

  answer: (session_id: string, question_id: string, answer: string | string[]) =>
    request<RecAnswerResponse>(`/recommendations/session/${session_id}/answer`, {
      method: 'POST',
      body: JSON.stringify({ question_id, answer }),
    }),

  results: (session_id: string) =>
    request<RecResultsResponse>(`/recommendations/session/${session_id}/results`),

  interaction: (recipe_id: number, interaction_type: string) =>
    request<null>('/recommendations/interaction', {
      method: 'POST',
      body: JSON.stringify({ recipe_id, interaction_type }),
    }),

  deleteInteraction: (recipe_id: number, interaction_type: string) =>
    request<null>(`/recommendations/interaction/${recipe_id}/${interaction_type}`, {
      method: 'DELETE',
    }),
};
