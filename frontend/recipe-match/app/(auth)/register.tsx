import React, { useEffect, useState } from 'react';
import {
  View, Text, TextInput, Pressable, StyleSheet,
  KeyboardAvoidingView, Platform, ScrollView, Alert, Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Link } from 'expo-router';
import { useAuthStore } from '@/store/authStore';
import { ApiError, DietaryProfile, recipesApi } from '@/services/api';
import { Colors } from '@/constants/colors';

type DietaryToggleKey = 'is_vegetarian' | 'is_vegan' | 'is_gluten_free' | 'is_dairy_free';

const DIETARY_OPTIONS: { key: DietaryToggleKey; label: string }[] = [
  { key: 'is_vegetarian', label: 'Vegetarian' },
  { key: 'is_vegan', label: 'Vegan' },
  { key: 'is_gluten_free', label: 'Gluten-free' },
  { key: 'is_dairy_free', label: 'Dairy-free' },
];

export default function RegisterScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [dietary, setDietary] = useState<DietaryProfile>({
    is_vegetarian: false,
    is_vegan: false,
    is_gluten_free: false,
    is_dairy_free: false,
    excluded_ingredients: [],
  });
  const [ingredientQuery, setIngredientQuery] = useState('');
  const [ingredientSuggestions, setIngredientSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const register = useAuthStore(s => s.register);

  useEffect(() => {
    const query = ingredientQuery.trim().toLowerCase();
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

  const toggle = (key: DietaryToggleKey) =>
    setDietary(d => ({ ...d, [key]: !d[key] }));

  const addExcludedIngredient = (value: string) => {
    const ingredient = value.trim().toLowerCase().replace(/\s+/g, ' ');
    if (!ingredient) return;
    setDietary(d => (
      d.excluded_ingredients.includes(ingredient)
        ? d
        : { ...d, excluded_ingredients: [...d.excluded_ingredients, ingredient] }
    ));
    setIngredientQuery('');
  };

  const removeExcludedIngredient = (value: string) => {
    setDietary(d => ({
      ...d,
      excluded_ingredients: d.excluded_ingredients.filter(item => item !== value),
    }));
  };

  const handleRegister = async () => {
    if (!email.trim() || !password) {
      Alert.alert('Missing fields', 'Please enter your email and password.');
      return;
    }
    if (password !== confirm) {
      Alert.alert('Password mismatch', 'The passwords you entered do not match.');
      return;
    }
    if (password.length < 8) {
      Alert.alert('Weak password', 'Password must be at least 8 characters.');
      return;
    }
    setLoading(true);
    try {
      await register(email.trim().toLowerCase(), password, dietary);
    } catch (err) {
      if (err instanceof ApiError && err.message === 'confirm_email') {
        Alert.alert(
          'Check your email',
          'We sent you a confirmation link. Click it to activate your account, then sign in.',
        );
        return;
      }
      const msg = err instanceof ApiError ? err.message : 'Registration failed';
      Alert.alert('Registration failed', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <View style={styles.logoMark}>
            <Ionicons name="restaurant" size={28} color={Colors.primary} />
          </View>
          <Text style={styles.kicker}>Recipe Match</Text>
          <Text style={styles.title}>Create account</Text>
          <Text style={styles.subtitle}>Tell us about your preferences</Text>
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={styles.input}
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
            autoComplete="email"
            placeholder="you@example.com"
            placeholderTextColor={Colors.textTertiary}
          />

          <Text style={styles.label}>Password</Text>
          <TextInput
            style={styles.input}
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            autoComplete="new-password"
            placeholder="At least 8 characters"
            placeholderTextColor={Colors.textTertiary}
          />

          <Text style={styles.label}>Confirm password</Text>
          <TextInput
            style={styles.input}
            value={confirm}
            onChangeText={setConfirm}
            secureTextEntry
            placeholder="Repeat password"
            placeholderTextColor={Colors.textTertiary}
          />

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
            style={[styles.btn, loading && styles.btnDisabled]}
            onPress={handleRegister}
            disabled={loading}
          >
            <Text style={styles.btnText}>{loading ? 'Creating account...' : 'Get started'}</Text>
          </Pressable>

          <View style={styles.footer}>
            <Text style={styles.footerText}>Already have an account? </Text>
            <Link href="/(auth)/login" asChild>
              <Pressable>
                <Text style={styles.link}>Sign in</Text>
              </Pressable>
            </Link>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: Colors.background },
  container: { flexGrow: 1, padding: 24, paddingTop: 52 },
  header: { marginBottom: 24 },
  logoMark: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
  },
  kicker: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  title: { fontSize: 32, fontWeight: '900', color: Colors.textPrimary },
  subtitle: { fontSize: 16, color: Colors.textSecondary, marginTop: 6, lineHeight: 22 },
  form: { gap: 8 },
  label: { fontSize: 14, fontWeight: '800', color: Colors.textPrimary, marginTop: 8 },
  input: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.hairline,
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    color: Colors.textPrimary,
    marginTop: 4,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '900',
    color: Colors.textPrimary,
    marginTop: 20,
    marginBottom: 4,
  },
  dietaryList: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.hairline,
    overflow: 'hidden',
  },
  dietaryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.hairline,
  },
  dietaryLabel: { fontSize: 15, color: Colors.textPrimary },
  ingredientBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.hairline,
    borderRadius: 14,
    marginTop: 4,
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
    backgroundColor: Colors.surfaceMuted,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  suggestionText: { color: Colors.textSecondary, fontSize: 13, fontWeight: '600' },
  excludedChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 10 },
  excludedChip: {
    backgroundColor: Colors.primarySoft,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  excludedChipText: { color: Colors.primaryDark, fontSize: 13, fontWeight: '700' },
  btn: {
    backgroundColor: Colors.primary,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 24,
  },
  btnDisabled: { opacity: 0.6 },
  btnText: { color: '#fff', fontSize: 17, fontWeight: '900' },
  footer: { flexDirection: 'row', justifyContent: 'center', marginTop: 24, marginBottom: 40 },
  footerText: { color: Colors.textSecondary, fontSize: 15 },
  link: { color: Colors.primary, fontSize: 15, fontWeight: '800' },
});
