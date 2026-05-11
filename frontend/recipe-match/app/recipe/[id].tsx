import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, Pressable,
  Image, Alert, Share, Modal, FlatList,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Colors } from '@/constants/colors';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { RecipeCard } from '@/components/RecipeCard';
import {
  recipesApi,
  savedApi,
  collectionsApi,
  RecipeDetail,
  Collection,
  SimilarRecipe,
  ApiError,
} from '@/services/api';
import { useAuthStore } from '@/store/authStore';

function parseLines(raw: string | null | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(/\||\n|;/)
    .map(s => s.trim())
    .filter(Boolean);
}

export default function RecipeDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const token = useAuthStore(s => s.token);
  const clearSession = useAuthStore(s => s.clearSession);

  const [recipe, setRecipe] = useState<RecipeDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [collectionModal, setCollectionModal] = useState(false);
  const [shoppingModal, setShoppingModal] = useState(false);
  const [shoppingList, setShoppingList] = useState<string[]>([]);
  const [similarModal, setSimilarModal] = useState(false);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarRecipes, setSimilarRecipes] = useState<SimilarRecipe[]>([]);

  useEffect(() => {
    const recipeId = Number(id);
    if (!id || isNaN(recipeId)) {
      setError('Invalid recipe');
      setLoading(false);
      return;
    }

    const requests: [Promise<RecipeDetail>, Promise<Collection[]>] = [
      recipesApi.get(recipeId),
      token ? collectionsApi.list() : Promise.resolve([]),
    ];

    Promise.all(requests)
      .then(([r, cols]) => {
        setRecipe(r);
        setCollections(cols);
      })
      .catch(err => {
        if (err instanceof ApiError && err.status === 401) {
          clearSession();
          return;
        }
        setError(err instanceof ApiError ? err.message : 'Failed to load recipe');
      })
      .finally(() => setLoading(false));
  }, [clearSession, id, token]);

  const handleSave = async (collectionId?: string) => {
    if (!recipe) return;
    try {
      await savedApi.save(recipe.id, collectionId);
      setSaved(true);
      setCollectionModal(false);
      Alert.alert('Saved', 'Recipe added to your collection.');
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Could not save recipe';
      Alert.alert('Error', msg);
    }
  };

  const handleUnsave = async () => {
    if (!recipe) return;
    try {
      await savedApi.unsave(recipe.id);
      setSaved(false);
    } catch {
      Alert.alert('Error', 'Could not remove recipe');
    }
  };

  const handleShoppingList = async () => {
    if (!recipe) return;
    try {
      const res = await recipesApi.shoppingList(recipe.id);
      setShoppingList(res.ingredients);
      setShoppingModal(true);
    } catch {
      Alert.alert('Error', 'Could not load shopping list');
    }
  };

  const handleShare = async () => {
    if (!recipe) return;
    await Share.share({ message: `Check out this recipe: ${recipe.name}` });
  };

  const handleSimilar = async () => {
    if (!recipe) return;
    setSimilarLoading(true);
    setSimilarModal(true);
    try {
      const res = await recipesApi.similar(recipe.id);
      setSimilarRecipes(res.recipes);
    } catch {
      Alert.alert('Error', 'Could not load similar recipes');
      setSimilarModal(false);
    } finally {
      setSimilarLoading(false);
    }
  };

  if (loading) return <LoadingSpinner fullScreen message="Loading recipe..." />;

  if (error || !recipe) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>{error ?? 'Recipe not found'}</Text>
        <Pressable style={styles.backBtn} onPress={() => router.back()}>
          <Text style={styles.backBtnText}>Go back</Text>
        </Pressable>
      </View>
    );
  }

  const badges: string[] = [];
  if (recipe.is_quick) badges.push('Quick');
  if (recipe.is_vegan) badges.push('Vegan');
  else if (recipe.is_vegetarian) badges.push('Vegetarian');
  if (recipe.is_gluten_free) badges.push('Gluten-free');
  if (recipe.is_dairy_free) badges.push('Dairy-free');
  if (recipe.is_spicy) badges.push('Spicy');
  if (recipe.is_sweet) badges.push('Sweet');

  return (
    <>
      <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
        <View style={styles.heroWrap}>
          {recipe.image_url ? (
            <Image source={{ uri: recipe.image_url }} style={styles.heroImage} />
          ) : (
            <View style={styles.heroPlaceholder}>
              <Ionicons name="restaurant-outline" size={44} color={Colors.textTertiary} />
            </View>
          )}
          <View style={styles.heroScrim} />
          <Pressable style={styles.backOverlay} onPress={() => router.back()}>
            <Ionicons name="chevron-back" size={24} color={Colors.textPrimary} />
          </Pressable>
        </View>

        <View style={styles.content}>
          <Text style={styles.name}>{recipe.name}</Text>

          <View style={styles.metaRow}>
            {recipe.total_minutes != null && (
              <View style={styles.metaChip}>
                <Ionicons name="time-outline" size={14} color={Colors.textSecondary} />
                <Text style={styles.metaChipText}>{recipe.total_minutes} min</Text>
              </View>
            )}
            {recipe.servings != null && (
              <View style={styles.metaChip}>
                <Ionicons name="people-outline" size={14} color={Colors.textSecondary} />
                <Text style={styles.metaChipText}>{recipe.servings} servings</Text>
              </View>
            )}
            {recipe.cuisine && (
              <View style={styles.metaChip}>
                <Ionicons name="earth-outline" size={14} color={Colors.textSecondary} />
                <Text style={styles.metaChipText}>
                  {recipe.cuisine.replace('_', ' ').charAt(0).toUpperCase() + recipe.cuisine.slice(1)}
                </Text>
              </View>
            )}
          </View>

          {badges.length > 0 && (
            <View style={styles.badges}>
              {badges.map(b => (
                <View key={b} style={styles.badge}>
                  <Text style={styles.badgeText}>{b}</Text>
                </View>
              ))}
            </View>
          )}

          {recipe.description && (
            <Text style={styles.description}>{recipe.description}</Text>
          )}

          <View style={styles.actions}>
            <Pressable
              style={[styles.actionBtn, saved && styles.actionBtnActive]}
              onPress={saved ? handleUnsave : () => setCollectionModal(true)}
            >
              <Ionicons
                name={saved ? 'heart' : 'heart-outline'}
                size={18}
                color={saved ? Colors.textInverse : Colors.primary}
              />
              <Text style={[styles.actionBtnText, saved && styles.actionBtnTextActive]}>
                {saved ? 'Saved' : 'Save'}
              </Text>
            </Pressable>
            <Pressable style={styles.actionBtnSecondary} onPress={handleShoppingList}>
              <Ionicons name="cart-outline" size={18} color={Colors.textPrimary} />
              <Text style={styles.actionBtnSecondaryText}>List</Text>
            </Pressable>
            <Pressable style={styles.actionBtnSecondary} onPress={handleShare}>
              <Ionicons name="share-outline" size={18} color={Colors.textPrimary} />
              <Text style={styles.actionBtnSecondaryText}>Share</Text>
            </Pressable>
          </View>

          <Pressable style={styles.similarBtn} onPress={handleSimilar}>
            <Ionicons name="sparkles" size={17} color={Colors.primaryDark} />
            <Text style={styles.similarBtnText}>Find more like this</Text>
          </Pressable>

          <Text style={styles.sectionTitle}>Ingredients</Text>
          {parseLines(recipe.ingredients).map((ing, i) => (
            <View key={i} style={styles.ingredientRow}>
              <View style={styles.bullet} />
              <Text style={styles.ingredientText}>{ing}</Text>
            </View>
          ))}

          {recipe.directions && (
            <>
              <Text style={styles.sectionTitle}>Directions</Text>
              {parseLines(recipe.directions).map((step, i) => (
                <View key={i} style={styles.stepRow}>
                  <View style={styles.stepNumber}>
                    <Text style={styles.stepNumberText}>{i + 1}</Text>
                  </View>
                  <Text style={styles.stepText}>{step}</Text>
                </View>
              ))}
            </>
          )}

          <View style={styles.bottomPad} />
        </View>
      </ScrollView>

      <Modal visible={collectionModal} transparent animationType="slide">
        <Pressable style={styles.overlay} onPress={() => setCollectionModal(false)}>
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>Save to folder</Text>
            <Pressable style={styles.sheetRow} onPress={() => handleSave()}>
              <Ionicons name="bookmark-outline" size={18} color={Colors.primary} />
              <Text style={styles.sheetRowText}>Default saved</Text>
            </Pressable>
            {collections.map(col => (
              <Pressable key={col.id} style={styles.sheetRow} onPress={() => handleSave(col.id)}>
                <Ionicons name="folder-outline" size={18} color={Colors.textSecondary} />
                <Text style={styles.sheetRowText}>{col.name}</Text>
              </Pressable>
            ))}
          </View>
        </Pressable>
      </Modal>

      <Modal visible={shoppingModal} transparent animationType="slide">
        <Pressable style={styles.overlay} onPress={() => setShoppingModal(false)}>
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>Shopping list</Text>
            <FlatList
              data={shoppingList}
              keyExtractor={(_, i) => String(i)}
              renderItem={({ item }) => (
                <View style={styles.ingredientRow}>
                  <View style={styles.bullet} />
                  <Text style={styles.ingredientText}>{item}</Text>
                </View>
              )}
              style={styles.sheetList}
            />
          </View>
        </Pressable>
      </Modal>

      <Modal visible={similarModal} transparent animationType="slide">
        <Pressable style={styles.overlay} onPress={() => setSimilarModal(false)}>
          <Pressable style={styles.sheet} onPress={() => {}}>
            <Text style={styles.sheetTitle}>More like this</Text>
            {similarLoading ? (
              <LoadingSpinner message="Finding similar recipes..." />
            ) : similarRecipes.length === 0 ? (
              <Text style={styles.emptySheetText}>No similar recipes found yet.</Text>
            ) : (
              <FlatList
                data={similarRecipes}
                keyExtractor={item => String(item.id)}
                renderItem={({ item }) => (
                  <RecipeCard
                    recipe={item}
                    onPress={() => {
                      setSimilarModal(false);
                      router.push(`/recipe/${item.id}`);
                    }}
                  />
                )}
                style={styles.similarList}
                showsVerticalScrollIndicator={false}
              />
            )}
          </Pressable>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  heroWrap: { height: 300, backgroundColor: Colors.surfaceAlt },
  heroImage: { width: '100%', height: '100%', resizeMode: 'cover' },
  heroPlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: Colors.surfaceMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroScrim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(35, 31, 26, 0.12)',
  },
  backOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    width: 42,
    height: 42,
    backgroundColor: 'rgba(255, 252, 248, 0.92)',
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    padding: 20,
    marginTop: -22,
    backgroundColor: Colors.background,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  name: { fontSize: 28, fontWeight: '900', color: Colors.textPrimary, lineHeight: 34 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 14 },
  metaChip: {
    backgroundColor: Colors.surface,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    borderWidth: 1,
    borderColor: Colors.hairline,
  },
  metaChipText: { fontSize: 13, color: Colors.textSecondary, fontWeight: '700' },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 12 },
  badge: {
    backgroundColor: Colors.primarySoft,
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
  },
  badgeText: { fontSize: 12, color: Colors.primaryDark, fontWeight: '800' },
  description: { fontSize: 15, color: Colors.textSecondary, lineHeight: 23, marginTop: 16 },
  actions: { flexDirection: 'row', gap: 10, marginTop: 22, marginBottom: 4 },
  actionBtn: {
    flex: 1,
    paddingVertical: 13,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 6,
    backgroundColor: Colors.surface,
  },
  actionBtnActive: { backgroundColor: Colors.primary },
  actionBtnText: { color: Colors.primary, fontWeight: '900', fontSize: 15 },
  actionBtnTextActive: { color: '#fff' },
  actionBtnSecondary: {
    paddingVertical: 13,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.hairline,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 5,
    backgroundColor: Colors.surface,
  },
  actionBtnSecondaryText: { color: Colors.textPrimary, fontWeight: '800', fontSize: 14 },
  similarBtn: {
    marginTop: 10,
    borderRadius: 16,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 13,
    flexDirection: 'row',
    gap: 7,
  },
  similarBtnText: { color: Colors.primaryDark, fontWeight: '900', fontSize: 15 },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '900',
    color: Colors.textPrimary,
    marginTop: 28,
    marginBottom: 12,
  },
  ingredientRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 9, gap: 10 },
  bullet: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: Colors.primary,
    marginTop: 8,
  },
  ingredientText: { flex: 1, fontSize: 15, color: Colors.textPrimary, lineHeight: 23 },
  stepRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 16, gap: 12 },
  stepNumber: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: Colors.textPrimary,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 1,
    flexShrink: 0,
  },
  stepNumberText: { color: '#fff', fontSize: 13, fontWeight: '900' },
  stepText: { flex: 1, fontSize: 15, color: Colors.textPrimary, lineHeight: 24 },
  bottomPad: { height: 40 },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 32, backgroundColor: Colors.background },
  errorText: { fontSize: 16, color: Colors.error, textAlign: 'center' },
  backBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 14,
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  backBtnText: { color: '#fff', fontWeight: '800' },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(35, 31, 26, 0.42)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 26,
    borderTopRightRadius: 26,
    padding: 24,
    maxHeight: '72%',
  },
  sheetTitle: { fontSize: 20, fontWeight: '900', color: Colors.textPrimary, marginBottom: 16 },
  sheetRow: {
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.hairline,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  sheetRowText: { fontSize: 16, color: Colors.textPrimary, fontWeight: '700' },
  sheetList: { maxHeight: 400 },
  similarList: { maxHeight: 460, marginHorizontal: -16 },
  emptySheetText: { color: Colors.textSecondary, fontSize: 15, lineHeight: 22 },
});
