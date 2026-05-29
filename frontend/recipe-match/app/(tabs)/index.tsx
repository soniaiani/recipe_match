import React, { useEffect } from 'react';
import {
  View, Text, StyleSheet, FlatList, Pressable,
  SafeAreaView, RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/colors';
import { RecipeCard } from '@/components/RecipeCard';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { useForYou } from '@/hooks/useForYou';
import { useSavedRecipes } from '@/hooks/useSavedRecipes';
import { useAuthStore } from '@/store/authStore';
import { RecipeSummary } from '@/services/api';

export default function ForYouScreen() {
  const router = useRouter();
  const user = useAuthStore(s => s.user);
  const logout = useAuthStore(s => s.logout);
  const {
    recipes, loading, error, load,
  } = useForYou();
  const {
    isSaved, saveRecipe, unsaveRecipe, loadAll,
  } = useSavedRecipes();

  useEffect(() => {
    load();
    loadAll();
  }, [load, loadAll]);

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
              onRefresh={load}
              tintColor={Colors.primary}
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
