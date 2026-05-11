import React from 'react';
import {
  FlatList, Pressable, StyleSheet, Text, View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/colors';
import { RecipeCard } from '@/components/RecipeCard';
import { ExplorerRecipe } from '@/services/explorerApi';
import { RecipeSummary } from '@/services/api';

type ExplorerResultsProps = {
  recipes: ExplorerRecipe[];
  selectedChain: string[];
  onBack: () => void;
};

function toRecipeSummary(recipe: ExplorerRecipe): RecipeSummary {
  return {
    id: recipe.id,
    name: recipe.name,
    description: recipe.description,
    image_url: recipe.image_url,
    meal_type: recipe.meal_type,
    cuisine: recipe.cuisine,
    total_minutes: null,
    is_vegetarian: Boolean(recipe.is_vegetarian),
    is_vegan: Boolean(recipe.is_vegan),
    is_gluten_free: Boolean(recipe.is_gluten_free),
    is_dairy_free: Boolean(recipe.is_dairy_free),
    is_quick: false,
  };
}

export function ExplorerResults({ recipes, selectedChain, onBack }: ExplorerResultsProps) {
  const router = useRouter();
  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <Pressable onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>← Back to Explorer</Text>
        </Pressable>
        <Text style={styles.title}>Ingredient matches</Text>
        <Text style={styles.subtitle}>Based on: {selectedChain.join(', ')}</Text>
      </View>

      <FlatList
        data={recipes}
        keyExtractor={item => String(item.id)}
        renderItem={({ item }) => (
          <RecipeCard
            recipe={toRecipeSummary(item)}
            onPress={() => router.push(`/recipe/${item.id}`)}
          />
        )}
        contentContainerStyle={styles.list}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: Colors.background },
  header: { paddingHorizontal: 20, paddingTop: 18, paddingBottom: 12 },
  backBtn: { alignSelf: 'flex-start', marginBottom: 12 },
  backText: { color: Colors.primary, fontWeight: '800', fontSize: 14 },
  title: { fontSize: 28, fontWeight: '900', color: Colors.textPrimary },
  subtitle: {
    marginTop: 6,
    color: Colors.textSecondary,
    fontSize: 14,
    fontWeight: '700',
  },
  list: { paddingTop: 8, paddingBottom: 28 },
});
