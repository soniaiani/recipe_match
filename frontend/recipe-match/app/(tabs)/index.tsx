import React, { useEffect } from 'react';
import {
  View, Text, StyleSheet, FlatList, Pressable,
  SafeAreaView, RefreshControl, Modal, ScrollView, Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/colors';
import { RecipeCard } from '@/components/RecipeCard';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { useForYou } from '@/hooks/useForYou';
import { useTasteProfile } from '@/hooks/useTasteProfile';
import { useTasteProfileRecipes } from '@/hooks/useTasteProfileRecipes';
import { useSavedRecipes } from '@/hooks/useSavedRecipes';
import { useAuthStore } from '@/store/authStore';
import {
  RecipeSummary, TasteProfileRecipesResponse, TasteProfileResponse,
} from '@/services/api';

function TasteProfileHeader({
  profile,
  loading,
  onOpen,
}: {
  profile: TasteProfileResponse | null;
  loading: boolean;
  onOpen: () => void;
}) {
  if (profile?.status === 'unavailable') return null;

  if (loading && !profile) {
    return (
      <View style={styles.tasteWrap}>
        <View style={styles.tasteTitleRow}>
          <Ionicons name="sparkles-outline" size={17} color={Colors.primary} />
          <Text style={styles.tasteEyebrow}>Your taste profile</Text>
        </View>
        <Text style={styles.tasteMuted}>Reading your recent recipe signals...</Text>
      </View>
    );
  }

  if (!profile || profile.status === 'insufficient_data') {
    return (
      <View style={styles.tasteWrap}>
        <View style={styles.tasteTitleRow}>
          <Ionicons name="sparkles-outline" size={17} color={Colors.primary} />
          <Text style={styles.tasteEyebrow}>Your taste profile</Text>
        </View>
        <Text style={styles.tasteMuted}>Explore, save, or cook a few recipes to shape this profile.</Text>
      </View>
    );
  }

  return (
    <Pressable style={styles.tasteWrap} onPress={onOpen}>
      <View style={styles.tasteTitleRow}>
        <Ionicons name="sparkles" size={17} color={Colors.primary} />
        <Text style={styles.tasteEyebrow}>Your taste profile</Text>
      </View>
      <Text style={styles.tasteSummary}>{profile.compact_summary}</Text>
      <View style={styles.tasteCards}>
        {profile.taste_cards.slice(0, 3).map(card => (
          <View key={card.title} style={styles.tasteCard}>
            <Text style={styles.tasteCardTitle}>{card.title}</Text>
            <Text style={styles.tasteCardText}>{card.text}</Text>
          </View>
        ))}
      </View>
      <View style={styles.expandRow}>
        <Text style={styles.expandText}>View details</Text>
        <Ionicons name="chevron-forward" size={15} color={Colors.primaryDark} />
      </View>
    </Pressable>
  );
}

function TasteProfileModal({
  profile,
  recommendations,
  recommendationsLoading,
  visible,
  onClose,
}: {
  profile: TasteProfileResponse | null;
  recommendations: TasteProfileRecipesResponse | null;
  recommendationsLoading: boolean;
  visible: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  if (!profile || profile.status !== 'ready') return null;
  const recommendationsByCluster = new Map(
    (recommendations?.clusters ?? []).map(cluster => [cluster.cluster_id, cluster.recipes]),
  );
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.overlay}>
        <Pressable style={styles.backdropHit} onPress={onClose} />
        <View style={styles.sheet}>
          <View style={styles.sheetHandle} />
          <Text style={styles.sheetTitle}>Your taste profile</Text>
          {profile.description && <Text style={styles.sheetDescription}>{profile.description}</Text>}
          <ScrollView showsVerticalScrollIndicator={false}>
            {profile.taste_cards.map((card, index) => {
              const clusterId = profile.top_clusters[index]?.cluster_id;
              const recipes = clusterId == null ? [] : recommendationsByCluster.get(clusterId) ?? [];
              return (
              <View key={`${card.title}-${clusterId ?? index}`} style={styles.clusterBlock}>
                <View style={styles.clusterHeader}>
                  <Text style={styles.clusterTitle}>{card.title}</Text>
                  <Text style={styles.clusterMeta}>{card.text}</Text>
                </View>
                {card.traits?.length > 0 && (
                  <View style={styles.traitSection}>
                    <Text style={styles.tasteRecipeEyebrow}>Taste traits</Text>
                    <View style={styles.traitChips}>
                      {card.traits.slice(0, 3).map(item => (
                        <View key={item} style={styles.traitChip}>
                          <Text style={styles.traitChipText}>{item}</Text>
                        </View>
                      ))}
                    </View>
                  </View>
                )}
                {card.ingredients.length > 0 && (
                  <View style={styles.ingredientChips}>
                    {card.ingredients.slice(0, 6).map(item => (
                      <View key={item} style={styles.ingredientChip}>
                        <Text style={styles.ingredientChipText}>{item}</Text>
                      </View>
                    ))}
                  </View>
                )}
                {recipes.length >= 2 && (
                  <View style={styles.tasteRecipeSection}>
                    <Text style={styles.tasteRecipeEyebrow}>Explore this taste</Text>
                    {recipes.map(recipe => (
                      <Pressable
                        key={recipe.id}
                        style={styles.tasteRecipeRow}
                        onPress={() => {
                          onClose();
                          router.push(`/recipe/${recipe.id}`);
                        }}
                      >
                        {recipe.image_url ? (
                          <Image source={{ uri: recipe.image_url }} style={styles.tasteRecipeImage} />
                        ) : (
                          <View style={styles.tasteRecipePlaceholder}>
                            <Ionicons name="restaurant-outline" size={18} color={Colors.textTertiary} />
                          </View>
                        )}
                        <View style={styles.tasteRecipeBody}>
                          <Text style={styles.tasteRecipeName} numberOfLines={2}>{recipe.name}</Text>
                          <Text style={styles.tasteRecipeMeta} numberOfLines={1}>
                            {[recipe.cuisine, recipe.total_minutes != null ? `${recipe.total_minutes} min` : null]
                              .filter(Boolean).join(' | ')}
                          </Text>
                        </View>
                        <Ionicons name="chevron-forward" size={17} color={Colors.textTertiary} />
                      </Pressable>
                    ))}
                  </View>
                )}
              </View>
              );
            })}
            {recommendationsLoading && (
              <View style={styles.tasteRecipeLoading}>
                <Ionicons name="sparkles-outline" size={16} color={Colors.primary} />
                <Text style={styles.tasteMuted}>Finding strong matches from your tastes...</Text>
              </View>
            )}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

export default function ForYouScreen() {
  const router = useRouter();
  const user = useAuthStore(s => s.user);
  const logout = useAuthStore(s => s.logout);
  const [tasteModalVisible, setTasteModalVisible] = React.useState(false);
  const {
    recipes, loading, error, load,
  } = useForYou();
  const {
    profile: tasteProfile,
    loading: tasteLoading,
    load: loadTasteProfile,
  } = useTasteProfile();
  const {
    data: tasteRecommendations,
    loading: tasteRecommendationsLoading,
    load: loadTasteRecommendations,
  } = useTasteProfileRecipes();
  const {
    isSaved, saveRecipe, unsaveRecipe, loadAll,
  } = useSavedRecipes();

  useEffect(() => {
    load();
    loadTasteProfile();
    loadAll();
  }, [load, loadAll, loadTasteProfile]);

  const refresh = () => {
    load();
    loadTasteProfile();
  };

  const openTasteProfile = () => {
    setTasteModalVisible(true);
    loadTasteRecommendations(recipes.map(recipe => recipe.id));
  };

  const handleToggleSave = async (recipe: RecipeSummary) => {
    try {
      if (isSaved(recipe.id)) {
        await unsaveRecipe(recipe.id);
      } else {
        await saveRecipe(recipe.id);
      }
    } catch {}
  };

  if (loading && recipes.length === 0) {
    return <LoadingSpinner fullScreen message="Finding recipes for you..." />;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>Today&apos;s picks</Text>
          <Text style={styles.heading}>For You</Text>
          <Text style={styles.subheading}>
            {user?.email ? `Tuned for ${user.email.split('@')[0]}` : 'Fresh ideas based on your taste'}
          </Text>
        </View>
        <Pressable onPress={logout} hitSlop={8} style={styles.logoutBtn}>
          <Ionicons name="log-out-outline" size={18} color={Colors.textSecondary} />
        </Pressable>
      </View>

      {error ? (
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryBtn} onPress={load}>
            <Text style={styles.retryText}>Try again</Text>
          </Pressable>
        </View>
      ) : (
        <FlatList
          data={recipes}
          keyExtractor={item => String(item.id)}
          initialNumToRender={20}
          maxToRenderPerBatch={20}
          windowSize={7}
          removeClippedSubviews={false}
          renderItem={({ item }) => (
            <RecipeCard
              recipe={item}
              onPress={() => router.push(`/recipe/${item.id}`)}
              onSave={() => handleToggleSave(item)}
              isSaved={isSaved(item.id)}
            />
          )}
          refreshControl={(
            <RefreshControl
              refreshing={loading}
              onRefresh={refresh}
              tintColor={Colors.primary}
            />
          )}
          ListHeaderComponent={(
            <TasteProfileHeader
              profile={tasteProfile}
              loading={tasteLoading}
              onOpen={openTasteProfile}
            />
          )}
          ListEmptyComponent={(
            <View style={styles.centered}>
              <View style={styles.emptyMark}>
                <Ionicons name="sparkles" size={28} color={Colors.primary} />
              </View>
              <Text style={styles.emptyTitle}>Save some recipes first</Text>
              <Text style={styles.emptySubtitle}>
                Use Find to discover recipes, then we&apos;ll personalize this feed.
              </Text>
            </View>
          )}
          ListFooterComponent={recipes.length > 0 ? (
            <Text style={styles.footerText}>{recipes.length} recommendations</Text>
          ) : null}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}
      <TasteProfileModal
        profile={tasteProfile}
        recommendations={tasteRecommendations}
        recommendationsLoading={tasteRecommendationsLoading}
        visible={tasteModalVisible}
        onClose={() => setTasteModalVisible(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 18,
  },
  greeting: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  heading: {
    fontSize: 32,
    fontWeight: '900',
    color: Colors.textPrimary,
    marginTop: 2,
  },
  subheading: { fontSize: 14, color: Colors.textSecondary, marginTop: 3 },
  logoutBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.hairline,
    marginBottom: 3,
  },
  list: { paddingBottom: 24 },
  tasteWrap: {
    paddingHorizontal: 20,
    paddingTop: 2,
    paddingBottom: 16,
  },
  tasteTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    marginBottom: 8,
  },
  tasteEyebrow: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  tasteSummary: {
    color: Colors.textPrimary,
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '800',
    marginBottom: 10,
  },
  tasteMuted: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 20,
  },
  tasteCards: {
    flexDirection: 'row',
    gap: 8,
  },
  tasteCard: {
    flex: 1,
    minHeight: 92,
    backgroundColor: Colors.surface,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.hairline,
    padding: 10,
  },
  tasteCardTitle: {
    color: Colors.textPrimary,
    fontSize: 12,
    fontWeight: '900',
    marginBottom: 5,
  },
  tasteCardText: {
    color: Colors.textSecondary,
    fontSize: 11,
    lineHeight: 15,
    fontWeight: '600',
  },
  expandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    marginTop: 10,
  },
  expandText: {
    color: Colors.primaryDark,
    fontSize: 13,
    fontWeight: '900',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(35, 31, 26, 0.35)',
    justifyContent: 'flex-end',
  },
  backdropHit: {
    flex: 1,
  },
  sheet: {
    maxHeight: '82%',
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 28,
  },
  sheetHandle: {
    alignSelf: 'center',
    width: 42,
    height: 4,
    borderRadius: 2,
    backgroundColor: Colors.border,
    marginBottom: 14,
  },
  sheetTitle: {
    color: Colors.textPrimary,
    fontSize: 24,
    fontWeight: '900',
  },
  sheetDescription: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    marginBottom: 14,
  },
  clusterBlock: {
    paddingVertical: 14,
    borderTopWidth: 1,
    borderTopColor: Colors.hairline,
  },
  clusterHeader: {
    marginBottom: 9,
  },
  clusterTitle: {
    color: Colors.textPrimary,
    fontSize: 16,
    fontWeight: '900',
  },
  clusterMeta: {
    color: Colors.textSecondary,
    fontSize: 14,
    fontWeight: '500',
    marginTop: 5,
    lineHeight: 20,
  },
  ingredientChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7,
  },
  traitSection: {
    marginBottom: 10,
    gap: 7,
  },
  traitChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7,
  },
  traitChip: {
    backgroundColor: Colors.accentSoft,
    borderRadius: 7,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  traitChipText: {
    color: Colors.accent,
    fontSize: 12,
    fontWeight: '800',
  },
  ingredientChip: {
    backgroundColor: Colors.primarySoft,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  ingredientChipText: {
    color: Colors.primaryDark,
    fontSize: 12,
    fontWeight: '800',
  },
  tasteRecipeSection: {
    marginTop: 13,
    gap: 8,
  },
  tasteRecipeEyebrow: {
    color: Colors.accent,
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  tasteRecipeRow: {
    minHeight: 64,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 7,
    borderTopWidth: 1,
    borderTopColor: Colors.hairline,
  },
  tasteRecipeImage: {
    width: 52,
    height: 52,
    borderRadius: 7,
  },
  tasteRecipePlaceholder: {
    width: 52,
    height: 52,
    borderRadius: 7,
    backgroundColor: Colors.surfaceMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tasteRecipeBody: {
    flex: 1,
    minWidth: 0,
  },
  tasteRecipeName: {
    color: Colors.textPrimary,
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '800',
  },
  tasteRecipeMeta: {
    color: Colors.textTertiary,
    fontSize: 11,
    marginTop: 3,
    textTransform: 'capitalize',
  },
  tasteRecipeLoading: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingVertical: 16,
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    gap: 10,
  },
  errorText: { color: Colors.error, fontSize: 15, textAlign: 'center' },
  retryBtn: {
    marginTop: 8,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  retryText: { color: '#fff', fontWeight: '800' },
  emptyMark: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: Colors.textPrimary,
    textAlign: 'center',
  },
  emptySubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
  },
  footerText: {
    color: Colors.textTertiary,
    fontSize: 12,
    fontWeight: '700',
    textAlign: 'center',
    paddingTop: 4,
    paddingBottom: 24,
  },
});
