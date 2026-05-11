import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors } from '@/constants/colors';
import { ApiError, DietaryProfile, recipesApi } from '@/services/api';
import { useAuthStore } from '@/store/authStore';

type DietaryToggleKey = 'is_vegetarian' | 'is_vegan' | 'is_gluten_free' | 'is_dairy_free';

const DIETARY_OPTIONS: { key: DietaryToggleKey; label: string }[] = [
  { key: 'is_vegetarian', label: 'Vegetarian' },
  { key: 'is_vegan', label: 'Vegan' },
  { key: 'is_gluten_free', label: 'Gluten-free' },
  { key: 'is_dairy_free', label: 'Dairy-free' },
];

const EMPTY_DIETARY: DietaryProfile = {
  is_vegetarian: false,
  is_vegan: false,
  is_gluten_free: false,
  is_dairy_free: false,
  excluded_ingredients: [],
};

function normalizeIngredient(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

export default function ProfileScreen() {
  const user = useAuthStore(s => s.user);
  const logout = useAuthStore(s => s.logout);
  const updateDietary = useAuthStore(s => s.updateDietary);
  const [dietary, setDietary] = useState<DietaryProfile>(user?.dietary ?? EMPTY_DIETARY);
  const [ingredientQuery, setIngredientQuery] = useState('');
  const [ingredientSuggestions, setIngredientSuggestions] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (user?.dietary) {
      setDietary(user.dietary);
    }
  }, [user?.dietary]);

  useEffect(() => {
    const query = normalizeIngredient(ingredientQuery);
    if (!query) {
      setIngredientSuggestions([]);
      return;
    }

    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const data = await recipesApi.ingredients(query, 12);
        if (!cancelled) {
          setIngredientSuggestions(
            data.ingredients.filter(item => !dietary.excluded_ingredients.includes(item)),
          );
        }
      } catch {
        if (!cancelled) {
          setIngredientSuggestions([]);
        }
      }
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ingredientQuery, dietary.excluded_ingredients]);

  const hasChanges = useMemo(
    () => JSON.stringify(dietary) !== JSON.stringify(user?.dietary ?? EMPTY_DIETARY),
    [dietary, user?.dietary],
  );

  const toggle = (key: DietaryToggleKey) => {
    setDietary(current => ({ ...current, [key]: !current[key] }));
  };

  const addExcludedIngredient = (value: string) => {
    const ingredient = normalizeIngredient(value);
    if (!ingredient) return;
    setDietary(current => (
      current.excluded_ingredients.includes(ingredient)
        ? current
        : { ...current, excluded_ingredients: [...current.excluded_ingredients, ingredient] }
    ));
    setIngredientQuery('');
  };

  const removeExcludedIngredient = (value: string) => {
    setDietary(current => ({
      ...current,
      excluded_ingredients: current.excluded_ingredients.filter(item => item !== value),
    }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateDietary(dietary);
      Alert.alert('Profile updated', 'Your preferences were saved.');
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Could not update profile.';
      Alert.alert('Update failed', message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          <View style={styles.header}>
            <View>
              <Text style={styles.eyebrow}>Account</Text>
              <Text style={styles.title}>Profile</Text>
            </View>
            <Pressable onPress={logout} hitSlop={8}>
              <Text style={styles.logoutText}>Sign out</Text>
            </Pressable>
          </View>

          <View style={styles.emailBox}>
            <Text style={styles.emailLabel}>Email</Text>
            <Text style={styles.emailText}>{user?.email ?? 'Signed in'}</Text>
          </View>

          <Text style={styles.sectionTitle}>Dietary preferences</Text>
          <View style={styles.dietaryList}>
            {DIETARY_OPTIONS.map(({ key, label }) => (
              <View key={key} style={styles.dietaryRow}>
                <Text style={styles.dietaryLabel}>{label}</Text>
                <Switch
                  value={dietary[key]}
                  onValueChange={() => toggle(key)}
                  trackColor={{ false: Colors.border, true: Colors.primaryLight }}
                  thumbColor={dietary[key] ? Colors.primary : Colors.textTertiary}
                />
              </View>
            ))}
          </View>

          <Text style={styles.sectionTitle}>Ingredients to avoid</Text>
          <View style={styles.ingredientBox}>
            <TextInput
              style={styles.ingredientInput}
              value={ingredientQuery}
              onChangeText={setIngredientQuery}
              placeholder="Search or type an ingredient"
              placeholderTextColor={Colors.textTertiary}
              autoCapitalize="none"
              onSubmitEditing={() => addExcludedIngredient(ingredientQuery)}
              returnKeyType="done"
            />
            {ingredientQuery.trim().length > 0 && (
              <Pressable
                style={styles.addIngredientBtn}
                onPress={() => addExcludedIngredient(ingredientQuery)}
              >
                <Text style={styles.addIngredientText}>Add</Text>
              </Pressable>
            )}
          </View>

          {ingredientQuery.trim().length > 0 && ingredientSuggestions.length > 0 && (
            <View style={styles.suggestions}>
              {ingredientSuggestions.map(item => (
                <Pressable
                  key={item}
                  style={styles.suggestionChip}
                  onPress={() => addExcludedIngredient(item)}
                >
                  <Text style={styles.suggestionText}>{item}</Text>
                </Pressable>
              ))}
            </View>
          )}

          {dietary.excluded_ingredients.length > 0 && (
            <View style={styles.excludedChips}>
              {dietary.excluded_ingredients.map(item => (
                <Pressable
                  key={item}
                  style={styles.excludedChip}
                  onPress={() => removeExcludedIngredient(item)}
                >
                  <Text style={styles.excludedChipText}>{item} x</Text>
                </Pressable>
              ))}
            </View>
          )}

          <Pressable
            style={[styles.saveBtn, (!hasChanges || saving) && styles.saveBtnDisabled]}
            onPress={handleSave}
            disabled={!hasChanges || saving}
          >
            <Text style={styles.saveText}>{saving ? 'Saving...' : 'Save changes'}</Text>
          </Pressable>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Colors.background },
  flex: { flex: 1 },
  container: { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 32 },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  eyebrow: { fontSize: 14, color: Colors.textSecondary },
  title: { fontSize: 28, fontWeight: '800', color: Colors.textPrimary, marginTop: 2 },
  logoutText: { fontSize: 14, color: Colors.textSecondary, marginBottom: 4 },
  emailBox: {
    backgroundColor: Colors.surface,
    borderWidth: 1.5,
    borderColor: Colors.border,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  emailLabel: { fontSize: 13, fontWeight: '700', color: Colors.textSecondary },
  emailText: { fontSize: 16, color: Colors.textPrimary, marginTop: 4 },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginTop: 20,
    marginBottom: 8,
  },
  dietaryList: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: Colors.border,
    overflow: 'hidden',
  },
  dietaryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  dietaryLabel: { fontSize: 15, color: Colors.textPrimary },
  ingredientBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderWidth: 1.5,
    borderColor: Colors.border,
    borderRadius: 12,
  },
  ingredientInput: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    color: Colors.textPrimary,
  },
  addIngredientBtn: {
    marginRight: 8,
    backgroundColor: Colors.primary,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  addIngredientText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  suggestions: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 8 },
  suggestionChip: {
    backgroundColor: Colors.surfaceAlt,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  suggestionText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  excludedChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 10 },
  excludedChip: {
    backgroundColor: Colors.primaryLight,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  excludedChipText: { color: Colors.primaryDark, fontSize: 13, fontWeight: '700' },
  saveBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 28,
  },
  saveBtnDisabled: { opacity: 0.55 },
  saveText: { color: '#fff', fontSize: 17, fontWeight: '700' },
});
