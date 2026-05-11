import React, {
  useEffect, useMemo, useRef, useState,
} from 'react';
import {
  FlatList, Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors } from '@/constants/colors';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { GraphCanvas } from '@/components/explorer/GraphCanvas';
import { ExplorerResults } from '@/components/explorer/ExplorerResults';
import {
  ExplorerRecipe, Suggestion, explorerApi,
} from '@/services/explorerApi';

type ExplorerState = 'SEARCH' | 'EXPLORING' | 'RESULTS';

export default function IngredientExplorerScreen() {
  const [explorerState, setExplorerState] = useState<ExplorerState>('SEARCH');
  const [inputValue, setInputValue] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [center, setCenter] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [selectedChain, setSelectedChain] = useState<string[]>([]);
  const [recipeCount, setRecipeCount] = useState(0);
  const [relaxed, setRelaxed] = useState(false);
  const [results, setResults] = useState<ExplorerRecipe[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchSeq = useRef(0);

  useEffect(() => {
    const query = inputValue.trim();
    const seq = searchSeq.current + 1;
    searchSeq.current = seq;

    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const res = await explorerApi.search(query);
        if (searchSeq.current === seq) setSearchResults(res.ingredients);
      } catch {
        if (searchSeq.current === seq) setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [inputValue]);

  const previousNodes = useMemo(
    () => selectedChain.slice(0, -1),
    [selectedChain],
  );

  const startWithIngredient = async (ingredient: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await explorerApi.start(ingredient);
      setCenter(res.center);
      setSuggestions(res.suggestions);
      setSelectedChain([res.center]);
      setRecipeCount(res.recipe_count);
      setRelaxed(false);
      setExplorerState('EXPLORING');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start explorer');
    } finally {
      setLoading(false);
    }
  };

  const expandIngredient = async (ingredient: string) => {
    if (loading) return;
    const nextChain = [...selectedChain, ingredient];
    setLoading(true);
    setError(null);
    try {
      const res = await explorerApi.expand(nextChain);
      setCenter(ingredient);
      setSelectedChain(nextChain);
      setSuggestions(res.suggestions);
      setRecipeCount(res.recipe_count);
      setRelaxed(res.relaxed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not expand ingredient');
    } finally {
      setLoading(false);
    }
  };

  const findRecipes = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await explorerApi.recommend(selectedChain);
      setResults(res.recipes);
      setRecipeCount(res.recipe_count);
      setRelaxed(res.relaxed);
      setExplorerState('RESULTS');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not find recipes');
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    if (explorerState === 'SEARCH') return;
    if (explorerState === 'RESULTS') {
      setExplorerState('EXPLORING');
      return;
    }
    setExplorerState('SEARCH');
    setInputValue('');
    setSearchResults([]);
    setCenter('');
    setSuggestions([]);
    setSelectedChain([]);
    setRecipeCount(0);
    setRelaxed(false);
  };

  if (loading && explorerState === 'SEARCH') {
    return <LoadingSpinner fullScreen message="Loading ingredients..." />;
  }

  if (explorerState === 'RESULTS') {
    return (
      <SafeAreaView style={styles.safe}>
        <ExplorerResults
          recipes={results}
          selectedChain={selectedChain}
          onBack={() => setExplorerState('EXPLORING')}
        />
      </SafeAreaView>
    );
  }

  if (explorerState === 'SEARCH') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.searchWrap}>
          <View style={styles.heroIcon}>
            <MaterialCommunityIcons name="graph" size={32} color={Colors.primary} />
          </View>
          <Text style={styles.title}>Ingredient Explorer</Text>
          <Text style={styles.subtitle}>Start with one ingredient and discover strong pairings.</Text>

          <TextInput
            value={inputValue}
            onChangeText={setInputValue}
            placeholder="Start with an ingredient..."
            placeholderTextColor={Colors.textTertiary}
            style={styles.input}
            autoCapitalize="none"
          />

          {error ? <Text style={styles.error}>{error}</Text> : null}

          <FlatList
            data={searchResults}
            keyExtractor={item => item}
            style={styles.autocomplete}
            keyboardShouldPersistTaps="handled"
            renderItem={({ item }) => (
              <Pressable style={styles.autocompleteItem} onPress={() => startWithIngredient(item)}>
                <Text style={styles.autocompleteText}>{item}</Text>
              </Pressable>
            )}
          />
        </View>
      </SafeAreaView>
    );
  }

  const lowCount = recipeCount < 10;
  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.topBar}>
        <Pressable onPress={goBack} style={styles.backBtn}>
          <Text style={styles.backText}>← Back</Text>
        </Pressable>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chain}>
          {selectedChain.map((item, index) => (
            <View key={`${item}-${index}`} style={styles.chainChip}>
              <Text style={styles.chainText}>{item}</Text>
            </View>
          ))}
        </ScrollView>
      </View>

      {error ? <Text style={styles.errorInline}>{error}</Text> : null}

      <GraphCanvas
        center={center}
        suggestions={suggestions}
        previousNodes={previousNodes}
        relaxed={relaxed}
        onNodePress={expandIngredient}
      />

      {relaxed ? <Text style={styles.relaxed}>Showing similar combinations</Text> : null}

      <View style={styles.bottomBar}>
        <View style={styles.countWrap}>
          <Text style={[styles.count, lowCount && styles.countLow]}>🍽 {recipeCount} recipes match</Text>
          {loading ? <Text style={styles.loadingText}>Updating graph...</Text> : null}
        </View>
        <Pressable
          style={[styles.findBtn, lowCount && styles.findBtnStrong, loading && styles.btnDisabled]}
          onPress={findRecipes}
          disabled={loading}
        >
          <Text style={styles.findText}>Find Recipes →</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Colors.background },
  searchWrap: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  heroIcon: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: 18,
  },
  title: {
    fontSize: 30,
    fontWeight: '900',
    color: Colors.textPrimary,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 15,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginTop: 8,
    marginBottom: 24,
  },
  input: {
    borderRadius: 18,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: Colors.surface,
    paddingHorizontal: 16,
    paddingVertical: 15,
    fontSize: 16,
    color: Colors.textPrimary,
    fontWeight: '700',
  },
  autocomplete: {
    maxHeight: 260,
    marginTop: 12,
    borderRadius: 16,
    backgroundColor: Colors.surface,
  },
  autocompleteItem: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: Colors.hairline,
  },
  autocompleteText: { fontSize: 15, color: Colors.textPrimary, fontWeight: '700' },
  error: { color: Colors.error, textAlign: 'center', marginTop: 12, fontWeight: '700' },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.hairline,
    gap: 10,
  },
  backBtn: { paddingVertical: 8, paddingRight: 4 },
  backText: { color: Colors.primary, fontWeight: '900' },
  chain: { gap: 8, alignItems: 'center' },
  chainChip: {
    paddingHorizontal: 11,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: Colors.primarySoft,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
  },
  chainText: { color: Colors.primaryDark, fontWeight: '800', fontSize: 12 },
  errorInline: {
    color: Colors.error,
    fontWeight: '700',
    textAlign: 'center',
    paddingTop: 8,
  },
  relaxed: {
    textAlign: 'center',
    color: Colors.textSecondary,
    fontWeight: '700',
    marginTop: -8,
  },
  bottomBar: {
    marginTop: 'auto',
    borderTopWidth: 1,
    borderTopColor: Colors.hairline,
    backgroundColor: Colors.surface,
    paddingHorizontal: 16,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  countWrap: { flex: 1 },
  count: { fontSize: 15, color: Colors.textPrimary, fontWeight: '900' },
  countLow: { color: Colors.warning },
  loadingText: { marginTop: 4, color: Colors.textSecondary, fontSize: 12, fontWeight: '700' },
  findBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  findBtnStrong: {
    backgroundColor: Colors.warning,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  btnDisabled: { opacity: 0.55 },
  findText: { color: Colors.textInverse, fontWeight: '900', fontSize: 14 },
});
