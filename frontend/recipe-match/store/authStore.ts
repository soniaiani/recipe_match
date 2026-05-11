import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authApi, ApiError, DietaryProfile, UserProfile } from '@/services/api';

const TOKEN_KEY = 'auth_token';

interface AuthState {
  token: string | null;
  user: UserProfile | null;
  initialized: boolean;
  initialize: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, dietary: DietaryProfile) => Promise<void>;
  logout: () => Promise<void>;
  clearSession: () => Promise<void>;
  updateDietary: (dietary: DietaryProfile) => Promise<void>;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: null,
  user: null,
  initialized: false,

  initialize: async () => {
    try {
      const stored = await AsyncStorage.getItem(TOKEN_KEY);
      if (stored) {
        const user = await authApi.me();
        set({ token: stored, user, initialized: true });
      } else {
        set({ initialized: true });
      }
    } catch {
      await AsyncStorage.removeItem(TOKEN_KEY);
      set({ token: null, user: null, initialized: true });
    }
  },

  login: async (email, password) => {
    const res = await authApi.login(email, password);
    await AsyncStorage.setItem(TOKEN_KEY, res.access_token);
    set({ token: res.access_token, user: res.user });
  },

  register: async (email, password, dietary) => {
    const res = await authApi.register(email, password, dietary);
    await AsyncStorage.setItem(TOKEN_KEY, res.access_token);
    set({ token: res.access_token, user: res.user });
  },

  logout: async () => {
    try {
      await authApi.logout();
    } catch {
      // ignore server errors on logout
    }
    await AsyncStorage.removeItem(TOKEN_KEY);
    set({ token: null, user: null });
  },

  clearSession: async () => {
    await AsyncStorage.removeItem(TOKEN_KEY);
    set({ token: null, user: null });
  },

  updateDietary: async (dietary) => {
    const updated = await authApi.updateDietary(dietary);
    set({ user: updated });
  },
}));
