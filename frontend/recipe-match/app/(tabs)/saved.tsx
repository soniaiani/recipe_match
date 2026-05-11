import React, { useCallback, useState } from 'react';
import {
  View, Text, StyleSheet, FlatList, Pressable,
  SafeAreaView, Alert, TextInput, Modal, ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { Colors } from '@/constants/colors';
import { RecipeCard } from '@/components/RecipeCard';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { useSavedRecipes } from '@/hooks/useSavedRecipes';
import { Collection } from '@/services/api';

export default function SavedScreen() {
  const router = useRouter();
  const {
    collections, visibleRecipes, activeCollectionId,
    loading, error,
    loadAll, unsaveRecipe, createCollection, deleteCollection, setActiveCollection,
  } = useSavedRecipes();

  const [newCollectionName, setNewCollectionName] = useState('');
  const [modalVisible, setModalVisible] = useState(false);

  useFocusEffect(
    useCallback(() => {
      loadAll();
    }, [loadAll]),
  );

  const handleCreateCollection = async () => {
    const name = newCollectionName.trim();
    if (!name) return;
    try {
      await createCollection(name);
      setNewCollectionName('');
      setModalVisible(false);
    } catch {
      Alert.alert('Error', 'Could not create collection');
    }
  };

  const handleDeleteCollection = (col: Collection) => {
    Alert.alert(
      `Delete "${col.name}"?`,
      'The recipes inside will not be deleted.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteCollection(col.id).catch(() =>
            Alert.alert('Error', 'Could not delete collection'),
          ),
        },
      ],
    );
  };

  const handleUnsave = (recipeId: number) => {
    Alert.alert('Remove recipe?', 'It will be removed from your saved list.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Remove',
        style: 'destructive',
        onPress: () => unsaveRecipe(recipeId).catch(() =>
          Alert.alert('Error', 'Could not remove recipe'),
        ),
      },
    ]);
  };

  if (loading && collections.length === 0) {
    return <LoadingSpinner fullScreen message="Loading saved recipes..." />;
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <View>
          <Text style={styles.eyebrow}>Collection</Text>
          <Text style={styles.heading}>Saved</Text>
        </View>
        <Pressable style={styles.addBtn} onPress={() => setModalVisible(true)}>
          <Ionicons name="add" size={17} color={Colors.primaryDark} />
          <Text style={styles.addBtnText}>Folder</Text>
        </Pressable>
      </View>

      <View style={styles.collectionTabsWrap}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.collectionTabsScroll}
          contentContainerStyle={styles.collectionTabs}
        >
          <Pressable
            style={[styles.collectionTab, !activeCollectionId && styles.collectionTabActive]}
            onPress={() => setActiveCollection(null)}
          >
            <Text style={[
              styles.collectionTabText,
              !activeCollectionId && styles.collectionTabTextActive,
            ]}
            >
              All
            </Text>
          </Pressable>
          {collections.map(col => (
            <Pressable
              key={col.id}
              style={[styles.collectionTab, activeCollectionId === col.id && styles.collectionTabActive]}
              onPress={() => setActiveCollection(col.id)}
              onLongPress={() => handleDeleteCollection(col)}
            >
              <Text style={[
                styles.collectionTabText,
                activeCollectionId === col.id && styles.collectionTabTextActive,
              ]}
              >
                {col.name}
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      {error ? (
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryBtn} onPress={loadAll}>
            <Text style={styles.retryText}>Try again</Text>
          </Pressable>
        </View>
      ) : (
        <FlatList
          data={visibleRecipes}
          keyExtractor={item => item.id}
          renderItem={({ item }) =>
            item.recipe ? (
              <RecipeCard
                recipe={item.recipe}
                onPress={() => router.push(`/recipe/${item.recipe!.id}`)}
                onSave={() => handleUnsave(item.recipe_id)}
                isSaved
              />
            ) : null}
          ListEmptyComponent={(
            <View style={styles.centered}>
              <View style={styles.emptyMark}>
                <Ionicons name="bookmark-outline" size={28} color={Colors.primary} />
              </View>
              <Text style={styles.emptyTitle}>Nothing saved yet</Text>
              <Text style={styles.emptySubtitle}>
                Tap the heart on any recipe to save it here.
              </Text>
            </View>
          )}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}

      <Modal
        visible={modalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setModalVisible(false)}
      >
        <Pressable style={styles.overlay} onPress={() => setModalVisible(false)}>
          <Pressable style={styles.dialog} onPress={() => {}}>
            <Text style={styles.dialogTitle}>New folder</Text>
            <TextInput
              style={styles.dialogInput}
              value={newCollectionName}
              onChangeText={setNewCollectionName}
              placeholder="e.g. Weekend meals"
              placeholderTextColor={Colors.textTertiary}
              autoFocus
              onSubmitEditing={handleCreateCollection}
            />
            <View style={styles.dialogActions}>
              <Pressable onPress={() => setModalVisible(false)} style={styles.dialogCancel}>
                <Text style={styles.dialogCancelText}>Cancel</Text>
              </Pressable>
              <Pressable onPress={handleCreateCollection} style={styles.dialogCreate}>
                <Text style={styles.dialogCreateText}>Create</Text>
              </Pressable>
            </View>
          </Pressable>
        </Pressable>
      </Modal>
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
    paddingBottom: 14,
  },
  eyebrow: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  heading: { fontSize: 32, fontWeight: '900', color: Colors.textPrimary, marginTop: 2 },
  addBtn: {
    backgroundColor: Colors.primarySoft,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
    marginBottom: 3,
  },
  addBtnText: { color: Colors.primaryDark, fontWeight: '800', fontSize: 13 },
  collectionTabsWrap: { height: 54 },
  collectionTabsScroll: { flexGrow: 0 },
  collectionTabs: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
    alignItems: 'center',
  },
  collectionTab: {
    minHeight: 36,
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  collectionTabActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primarySoft,
  },
  collectionTabText: { fontSize: 14, color: Colors.textSecondary, fontWeight: '700' },
  collectionTabTextActive: { color: Colors.primaryDark, fontWeight: '900' },
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
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingHorizontal: 20,
    paddingVertical: 10,
    marginTop: 8,
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
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(35, 31, 26, 0.42)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  dialog: {
    backgroundColor: Colors.surface,
    borderRadius: 22,
    padding: 24,
    width: '100%',
    maxWidth: 360,
    gap: 16,
  },
  dialogTitle: { fontSize: 19, fontWeight: '900', color: Colors.textPrimary },
  dialogInput: {
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surfaceMuted,
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    color: Colors.textPrimary,
  },
  dialogActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 12 },
  dialogCancel: { paddingHorizontal: 16, paddingVertical: 10 },
  dialogCancelText: { color: Colors.textSecondary, fontWeight: '700' },
  dialogCreate: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  dialogCreateText: { color: '#fff', fontWeight: '800' },
});
