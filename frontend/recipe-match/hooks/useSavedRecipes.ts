import { useState, useCallback } from 'react';
import { savedApi, collectionsApi, SavedRecipe, Collection, ApiError } from '@/services/api';
import { useAuthStore } from '@/store/authStore';

interface SavedState {
  collections: Collection[];
  savedRecipes: SavedRecipe[];
  activeCollectionId: string | null;
  loading: boolean;
  error: string | null;
}

export function useSavedRecipes() {
  const token = useAuthStore(s => s.token);
  const clearSession = useAuthStore(s => s.clearSession);
  const [state, setState] = useState<SavedState>({
    collections: [],
    savedRecipes: [],
    activeCollectionId: null,
    loading: false,
    error: null,
  });

  const loadAll = useCallback(async () => {
    if (!token) {
      setState(s => ({
        ...s,
        collections: [],
        savedRecipes: [],
        activeCollectionId: null,
        loading: false,
        error: null,
      }));
      return;
    }

    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const [collections, savedRecipes] = await Promise.all([
        collectionsApi.list(),
        savedApi.list(),
      ]);
      setState(s => ({
        ...s,
        collections,
        savedRecipes,
        loading: false,
      }));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        await clearSession();
        setState(s => ({
          ...s,
          collections: [],
          savedRecipes: [],
          activeCollectionId: null,
          loading: false,
          error: null,
        }));
        return;
      }
      const msg = err instanceof ApiError ? err.message : 'Failed to load saved recipes';
      setState(s => ({ ...s, loading: false, error: msg }));
    }
  }, [clearSession, token]);

  const saveRecipe = useCallback(async (recipeId: number, collectionId?: string) => {
    if (!token) {
      throw new Error('You must be logged in to save recipes');
    }
    try {
      await savedApi.save(recipeId, collectionId);
      await loadAll();
    } catch (err) {
      throw err instanceof ApiError ? err : new Error('Failed to save recipe');
    }
  }, [loadAll, token]);

  const unsaveRecipe = useCallback(async (recipeId: number) => {
    if (!token) {
      throw new Error('You must be logged in to remove saved recipes');
    }
    try {
      await savedApi.unsave(recipeId);
      setState(s => ({
        ...s,
        savedRecipes: s.savedRecipes.filter(r => r.recipe_id !== recipeId),
      }));
    } catch (err) {
      throw err instanceof ApiError ? err : new Error('Failed to unsave recipe');
    }
  }, [token]);

  const createCollection = useCallback(async (name: string) => {
    if (!token) {
      throw new Error('You must be logged in to create collections');
    }
    try {
      await collectionsApi.create(name);
      await loadAll();
    } catch (err) {
      throw err instanceof ApiError ? err : new Error('Failed to create collection');
    }
  }, [loadAll, token]);

  const deleteCollection = useCallback(async (id: string) => {
    if (!token) {
      throw new Error('You must be logged in to delete collections');
    }
    try {
      await collectionsApi.delete(id);
      setState(s => ({
        ...s,
        collections: s.collections.filter(c => c.id !== id),
        activeCollectionId: s.activeCollectionId === id ? null : s.activeCollectionId,
        savedRecipes: s.savedRecipes.filter(r => r.collection_id !== id),
      }));
    } catch (err) {
      throw err instanceof ApiError ? err : new Error('Failed to delete collection');
    }
  }, [token]);

  const setActiveCollection = useCallback((id: string | null) => {
    setState(s => ({ ...s, activeCollectionId: id }));
  }, []);

  const isSaved = useCallback(
    (recipeId: number) => state.savedRecipes.some(r => r.recipe_id === recipeId),
    [state.savedRecipes],
  );

  const visibleRecipes = state.activeCollectionId
    ? state.savedRecipes.filter(r => r.collection_id === state.activeCollectionId)
    : state.savedRecipes;

  return {
    ...state,
    visibleRecipes,
    loadAll,
    saveRecipe,
    unsaveRecipe,
    createCollection,
    deleteCollection,
    setActiveCollection,
    isSaved,
  };
}
