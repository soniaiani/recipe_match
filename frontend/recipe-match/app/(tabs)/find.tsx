import React, { useRef, useState } from 'react';
import {
  View, Text, StyleSheet, Pressable, FlatList,
  ScrollView, Dimensions, TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  runOnJS,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { Image } from 'expo-image';
import { Colors } from '@/constants/colors';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { useRecommendation } from '@/hooks/useRecommendation';
import { useAuthStore } from '@/store/authStore';
import { RecQuestion, RecScoredRecipe } from '@/services/api';

const SWIPE_THRESHOLD = 100;
const SCREEN_WIDTH = Dimensions.get('window').width;

const QUESTION_LABELS: Record<string, string> = {
  is_spicy: 'Should the recipe taste spicy?',
  is_sweet: 'Should the recipe taste sweet?',
  is_quick: 'Do you want a quick recipe?',
  needs_oven: 'Can the recipe use an oven?',
  needs_stovetop: 'Can the recipe use the stovetop?',
  is_no_cook: 'Do you want a no-cook recipe?',
  has_pasta: 'Should it include pasta?',
  has_rice: 'Should it include rice?',
  has_potato: 'Should it include potatoes?',
  has_tomato_base: 'Should it have a tomato base?',
  has_cream_base: 'Should it be creamy?',
  has_cheese: 'Should it include cheese?',
  has_broth_base: 'Should it be broth-based?',
  has_mushroom: 'Should it include mushrooms?',
  has_leafy_greens: 'Should it include leafy greens?',
  has_beans_legumes: 'Should it include beans or legumes?',
  has_fruit: 'Should it include fruit?',
  has_nuts: 'Should it include nuts?',
  has_chocolate: 'Should it include chocolate?',
  has_tortilla: 'Should it include tortillas or tacos?',
  has_asian_sauce: 'Should it include an Asian-style sauce?',
};

const OPTION_LABELS: Record<string, Record<string, string>> = {
  meal_type: {
    appetizer: 'Appetizer',
    breakfast: 'Breakfast',
    dessert: 'Dessert',
    drink: 'Drink',
    lunch_dinner: 'Lunch / Dinner',
    salad_side: 'Salad / Side',
    snack: 'Snack',
    soup: 'Soup',
    condiment: 'Condiment',
  },
  protein_type: {
    chicken: 'Chicken',
    beef_pork: 'Beef / Pork',
    fish_seafood: 'Fish / Seafood',
    meatless: 'Meatless',
  },
  cuisine: {
    italian: 'Italian',
    asian: 'Asian',
    mexican: 'Mexican',
    french: 'French',
    mediterranean: 'Mediterranean',
    indian: 'Indian',
    american: 'American',
    other: 'Other',
  },
};

const QUESTION_HEADINGS: Record<string, string> = {
  meal_type: 'What kind of meal are you looking for?',
  protein_type: 'Which protein sounds best?',
  cuisine: 'Which cuisine are you in the mood for?',
};

const SELECTION_QUESTION_IDS = new Set(['meal_type', 'protein_type', 'cuisine']);
const CRAVING_SUGGESTIONS = [
  'Creamy pasta',
  'Spicy soup',
  'Fresh salad',
  'Chocolate dessert',
  'Quick dinner',
];

function optionLabel(questionId: string, value: string): string {
  return OPTION_LABELS[questionId]?.[value] ?? value.replace(/_/g, ' ');
}

function isSelectionQuestion(question: RecQuestion): boolean {
  return (
    SELECTION_QUESTION_IDS.has(question.id) ||
    question.type === 'categorical' ||
    question.type === 'multiselect'
  );
}

function questionOptions(question: RecQuestion): string[] {
  return question.options ?? Object.keys(OPTION_LABELS[question.id] ?? {});
}

function allowsMultipleSelection(question: RecQuestion): boolean {
  return question.type === 'multiselect' || question.id === 'protein_type' || question.id === 'cuisine';
}

interface SwipeCardProps {
  question: RecQuestion;
  onAnswer: (answer: string) => void;
}

function SwipeCard({ question, onAnswer }: SwipeCardProps) {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const answered = useRef(false);
  const label = QUESTION_LABELS[question.id] ?? question.id.replace(/_/g, ' ');

  const pan = Gesture.Pan()
    .onUpdate(e => {
      translateX.value = e.translationX;
      translateY.value = e.translationY * 0.3;
    })
    .onEnd(e => {
      if (answered.current) return;
      if (e.translationX > SWIPE_THRESHOLD) {
        answered.current = true;
        translateX.value = withTiming(SCREEN_WIDTH * 1.5, { duration: 300 }, () => {
          runOnJS(onAnswer)('yes');
        });
      } else if (e.translationX < -SWIPE_THRESHOLD) {
        answered.current = true;
        translateX.value = withTiming(-SCREEN_WIDTH * 1.5, { duration: 300 }, () => {
          runOnJS(onAnswer)('no');
        });
      } else {
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
      }
    });

  const cardStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { rotate: `${interpolate(translateX.value, [-200, 200], [-12, 12], Extrapolation.CLAMP)}deg` },
    ],
  }));

  const yesOverlayStyle = useAnimatedStyle(() => ({
    opacity: interpolate(translateX.value, [0, SWIPE_THRESHOLD], [0, 0.9], Extrapolation.CLAMP),
  }));

  const noOverlayStyle = useAnimatedStyle(() => ({
    opacity: interpolate(translateX.value, [-SWIPE_THRESHOLD, 0], [0.9, 0], Extrapolation.CLAMP),
  }));

  return (
    <GestureDetector gesture={pan}>
      <Animated.View style={[styles.card, cardStyle]}>
        <Animated.View style={[styles.cardOverlay, styles.yesOverlay, yesOverlayStyle]}>
          <Ionicons name="checkmark-circle" size={42} color={Colors.textInverse} />
          <Text style={styles.overlayText}>YES</Text>
        </Animated.View>
        <Animated.View style={[styles.cardOverlay, styles.noOverlay, noOverlayStyle]}>
          <Ionicons name="close-circle" size={42} color={Colors.textInverse} />
          <Text style={styles.overlayText}>NO</Text>
        </Animated.View>

        <View style={styles.cardIcon}>
          <Ionicons name="restaurant-outline" size={28} color={Colors.primary} />
        </View>
        <Text style={styles.cardQuestion}>{label}</Text>
        <Text style={styles.cardHint}>Swipe the card or use the buttons</Text>
      </Animated.View>
    </GestureDetector>
  );
}

export default function FindScreen() {
  const router = useRouter();
  const user = useAuthStore(s => s.user);
  const {
    phase, currentQuestion, progress,
    results, resultsCount, loading, error, startSession, submitAnswer, reset,
  } = useRecommendation();

  const [selected, setSelected] = useState<string[]>([]);
  const [cardKey, setCardKey] = useState(0);
  const [semanticQuery, setSemanticQuery] = useState('');

  const handleStart = () => {
    setSelected([]);
    startSession(user?.dietary, semanticQuery);
  };

  const handleSuggestionPress = (value: string) => {
    setSemanticQuery(value);
  };

  const handleOptionPress = (value: string, question: RecQuestion) => {
    if (!allowsMultipleSelection(question)) {
      setSelected([value]);
      return;
    }
    setSelected(prev =>
      prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value],
    );
  };

  const handleSelectAny = () => setSelected(['any']);

  const handleSubmitFixed = async () => {
    if (!currentQuestion || selected.length === 0) return;
    const answer = allowsMultipleSelection(currentQuestion) ? selected : selected[0];
    setSelected([]);
    await submitAnswer(currentQuestion.id, answer);
    setCardKey(k => k + 1);
  };

  const handleSwipeAnswer = async (answer: string) => {
    if (!currentQuestion) return;
    await submitAnswer(currentQuestion.id, answer);
    setCardKey(k => k + 1);
  };

  const handleUnknown = async () => {
    if (!currentQuestion) return;
    await submitAnswer(currentQuestion.id, 'skip');
    setCardKey(k => k + 1);
  };

  if (phase === 'idle') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.hero}>
          <View style={styles.heroBadge}>
            <Ionicons name="search" size={26} color={Colors.primary} />
          </View>
          <Text style={styles.heroKicker}>Smart picker</Text>
          <Text style={styles.heroTitle}>Find the right recipe</Text>
          <Text style={styles.heroSubtitle}>
            Describe a dish, ingredient, or mood, then answer a few quick questions.
          </Text>
          <View style={styles.cravingBox}>
            <Text style={styles.cravingLabel}>What are you craving?</Text>
            <TextInput
              value={semanticQuery}
              onChangeText={setSemanticQuery}
              placeholder="Creamy chicken pasta, spicy soup..."
              placeholderTextColor={Colors.textTertiary}
              style={styles.cravingInput}
              returnKeyType="done"
              maxLength={120}
            />
            <View style={styles.suggestionRow}>
              {CRAVING_SUGGESTIONS.map(item => (
                <Pressable
                  key={item}
                  style={[
                    styles.suggestionChip,
                    semanticQuery === item && styles.suggestionChipActive,
                  ]}
                  onPress={() => handleSuggestionPress(item)}
                >
                  <Text
                    style={[
                      styles.suggestionText,
                      semanticQuery === item && styles.suggestionTextActive,
                    ]}
                  >
                    {item}
                  </Text>
                </Pressable>
              ))}
            </View>
          </View>
          <Pressable style={styles.startBtn} onPress={handleStart}>
            <Text style={styles.startBtnText}>Start search</Text>
            <Ionicons name="arrow-forward" size={18} color={Colors.textInverse} />
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  if (phase === 'error') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.hero}>
          <View style={[styles.heroBadge, styles.errorBadge]}>
            <Ionicons name="alert-circle-outline" size={28} color={Colors.error} />
          </View>
          <Text style={styles.heroTitle}>Something went wrong</Text>
          <Text style={styles.heroSubtitle}>{error}</Text>
          <Pressable style={styles.startBtn} onPress={reset}>
            <Text style={styles.startBtnText}>Try again</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  if (phase === 'done') {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.resultsHeader}>
          <View>
            <Text style={styles.greeting}>Best matches</Text>
            <Text style={styles.resultsTitle}>Top recipes for you</Text>
          </View>
          <Pressable onPress={reset} style={styles.restartBtn}>
            <Ionicons name="refresh" size={16} color={Colors.primary} />
            <Text style={styles.restartText}>Restart</Text>
          </Pressable>
        </View>

        {resultsCount === 0 ? (
          <View style={styles.hero}>
            <View style={styles.heroBadge}>
              <Ionicons name="search-outline" size={28} color={Colors.primary} />
            </View>
            <Text style={styles.heroTitle}>No recipes found</Text>
            <Text style={styles.heroSubtitle}>
              Try again with broader preferences for better results.
            </Text>
            <Pressable style={styles.startBtn} onPress={reset}>
              <Text style={styles.startBtnText}>Try again</Text>
            </Pressable>
          </View>
        ) : (
          <FlatList
            data={results}
            keyExtractor={item => String(item.id)}
            numColumns={2}
            columnWrapperStyle={styles.resultRow}
            renderItem={({ item }) => (
              <ResultCard recipe={item} onPress={() => router.push(`/recipe/${item.id}`)} />
            )}
            contentContainerStyle={styles.resultsList}
            showsVerticalScrollIndicator={false}
            ListFooterComponent={
              resultsCount < 5 ? (
                <View style={styles.specificNote}>
                  <Ionicons name="filter-outline" size={16} color={Colors.primaryDark} />
                  <Text style={styles.specificNoteText}>
                    Your preferences are very specific. These are the closest matches.
                  </Text>
                </View>
              ) : null
            }
          />
        )}
      </SafeAreaView>
    );
  }

  if (!currentQuestion) {
    return <LoadingSpinner fullScreen message="Calculating..." />;
  }

  const pct = Math.min((progress.current / progress.max) * 100, 100);

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.progressContainer}>
        <View style={styles.progressRow}>
          <Text style={styles.progressText}>
            Question {progress.current} of {progress.max}
          </Text>
          <Text style={styles.progressPercent}>{Math.round(pct)}%</Text>
        </View>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${pct}%` }]} />
        </View>
      </View>

      {!isSelectionQuestion(currentQuestion) && currentQuestion.type === 'boolean' && (
        <View style={styles.swipeContainer}>
          <SwipeCard
            key={cardKey}
            question={currentQuestion}
            onAnswer={handleSwipeAnswer}
          />
          <View style={styles.swipeHints}>
            <Pressable style={[styles.answerBtn, styles.answerNo]} onPress={() => handleSwipeAnswer('no')} disabled={loading}>
              <Ionicons name="close" size={18} color={Colors.error} />
              <Text style={[styles.answerText, styles.answerNoText]}>No</Text>
            </Pressable>
            <Pressable style={styles.skipBtn} onPress={handleUnknown} disabled={loading}>
              <Text style={styles.skipText}>Not sure</Text>
            </Pressable>
            <Pressable style={[styles.answerBtn, styles.answerYes]} onPress={() => handleSwipeAnswer('yes')} disabled={loading}>
              <Ionicons name="checkmark" size={18} color={Colors.success} />
              <Text style={[styles.answerText, styles.answerYesText]}>Yes</Text>
            </Pressable>
          </View>
        </View>
      )}

      {isSelectionQuestion(currentQuestion) && (
        <ScrollView contentContainerStyle={styles.chipContainer} bounces={false}>
          <Text style={styles.chipKicker}>Preferences</Text>
          <Text style={styles.chipHeading}>
            {QUESTION_HEADINGS[currentQuestion.id] ?? currentQuestion.id.replace(/_/g, ' ')}
          </Text>

          <View style={styles.chipGrid}>
            {questionOptions(currentQuestion).map(opt => {
              const active = selected.includes(opt);
              return (
                <Pressable
                  key={opt}
                  style={[styles.chip, active && styles.chipActive]}
                  onPress={() => handleOptionPress(opt, currentQuestion)}
                >
                  <Text style={[styles.chipText, active && styles.chipTextActive]}>
                    {optionLabel(currentQuestion.id, opt)}
                  </Text>
                </Pressable>
              );
            })}
          </View>

          {allowsMultipleSelection(currentQuestion) && (
            <Pressable
              style={[styles.anyBtn, selected.length === 1 && selected[0] === 'any' && styles.anyBtnActive]}
              onPress={handleSelectAny}
            >
              <Ionicons name="options-outline" size={17} color={Colors.textSecondary} />
              <Text style={styles.anyBtnText}>Any / No preference</Text>
            </Pressable>
          )}

          <Pressable
            style={[styles.nextBtn, selected.length === 0 && styles.nextBtnDisabled]}
            onPress={handleSubmitFixed}
            disabled={selected.length === 0 || loading}
          >
            <Text style={styles.nextBtnText}>
              {loading ? 'Calculating...' : 'Continue'}
            </Text>
          </Pressable>
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

function matchScoreColor(score: number): string {
  if (score >= 80) return Colors.success;
  if (score >= 60) return Colors.warning;
  return Colors.primary;
}

function ResultCard({
  recipe,
  onPress,
}: {
  recipe: RecScoredRecipe;
  onPress: () => void;
}) {
  const badgeColor = matchScoreColor(recipe.match_score);
  return (
    <Pressable style={styles.resultCard} onPress={onPress}>
      <Image
        source={recipe.image_url ?? undefined}
        style={styles.resultImage}
        contentFit="cover"
        placeholder={{ blurhash: 'L6Pj0^jE.AyE_3t7t7R**0o#DgR4' }}
      />
      <View style={[styles.matchBadge, { backgroundColor: badgeColor }]}>
        <Text style={styles.matchBadgeText}>{recipe.match_score}%</Text>
      </View>
      <View style={styles.resultInfo}>
        <Text style={styles.resultName} numberOfLines={2}>{recipe.name}</Text>
        {recipe.cuisine ? (
          <Text style={styles.resultMeta}>{optionLabel('cuisine', recipe.cuisine)}</Text>
        ) : null}
        {recipe.meal_type ? (
          <Text style={styles.resultMeta}>{optionLabel('meal_type', recipe.meal_type)}</Text>
        ) : null}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Colors.background },
  hero: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  heroBadge: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 18,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
  },
  errorBadge: { backgroundColor: Colors.errorSoft, borderColor: Colors.errorSoft },
  heroKicker: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  heroTitle: {
    fontSize: 30,
    fontWeight: '900',
    color: Colors.textPrimary,
    textAlign: 'center',
    lineHeight: 36,
  },
  heroSubtitle: {
    fontSize: 15,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
    marginTop: 10,
    maxWidth: 340,
  },
  cravingBox: {
    width: '100%',
    maxWidth: 380,
    marginTop: 24,
    padding: 14,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surface,
  },
  cravingLabel: {
    fontSize: 13,
    color: Colors.textPrimary,
    fontWeight: '900',
    marginBottom: 10,
  },
  cravingInput: {
    minHeight: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: Colors.surfaceMuted,
    paddingHorizontal: 14,
    fontSize: 15,
    color: Colors.textPrimary,
    fontWeight: '700',
  },
  suggestionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 12,
  },
  suggestionChip: {
    paddingHorizontal: 11,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surfaceElevated,
  },
  suggestionChipActive: {
    borderColor: Colors.primary,
    backgroundColor: Colors.primarySoft,
  },
  suggestionText: {
    fontSize: 12,
    color: Colors.textSecondary,
    fontWeight: '800',
  },
  suggestionTextActive: { color: Colors.primaryDark },
  startBtn: {
    marginTop: 24,
    backgroundColor: Colors.primary,
    borderRadius: 16,
    paddingHorizontal: 24,
    paddingVertical: 15,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.16,
    shadowRadius: 16,
    elevation: 5,
  },
  startBtnText: { color: '#fff', fontSize: 16, fontWeight: '800' },
  progressContainer: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 4 },
  progressRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  progressText: { fontSize: 13, color: Colors.textSecondary, fontWeight: '700' },
  progressPercent: { fontSize: 13, color: Colors.primaryDark, fontWeight: '900' },
  progressBar: {
    height: 6,
    backgroundColor: Colors.surfaceAlt,
    borderRadius: 999,
    overflow: 'hidden',
  },
  progressFill: { height: '100%', backgroundColor: Colors.primary, borderRadius: 999 },
  swipeContainer: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  card: {
    width: SCREEN_WIDTH - 48,
    aspectRatio: 0.95,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: Colors.hairline,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.14,
    shadowRadius: 28,
    elevation: 8,
    overflow: 'hidden',
  },
  cardOverlay: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  yesOverlay: { backgroundColor: Colors.success },
  noOverlay: { backgroundColor: Colors.error },
  overlayText: { fontSize: 32, fontWeight: '900', color: '#fff' },
  cardIcon: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: Colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 22,
  },
  cardQuestion: {
    fontSize: 27,
    fontWeight: '900',
    color: Colors.textPrimary,
    textAlign: 'center',
    lineHeight: 34,
  },
  cardHint: { marginTop: 18, fontSize: 13, color: Colors.textSecondary, fontWeight: '600' },
  swipeHints: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 24,
    gap: 12,
  },
  answerBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 13,
    borderRadius: 16,
    borderWidth: 1,
  },
  answerNo: { backgroundColor: Colors.errorSoft, borderColor: Colors.errorSoft },
  answerYes: { backgroundColor: Colors.successSoft, borderColor: Colors.successSoft },
  answerText: { fontSize: 15, fontWeight: '900' },
  answerNoText: { color: Colors.error },
  answerYesText: { color: Colors.success },
  skipBtn: {
    flex: 1.15,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
    paddingVertical: 13,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surface,
  },
  skipText: { fontSize: 14, color: Colors.textSecondary, fontWeight: '800' },
  chipContainer: { padding: 20, paddingBottom: 32 },
  chipKicker: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  chipHeading: {
    fontSize: 26,
    fontWeight: '900',
    color: Colors.textPrimary,
    marginBottom: 22,
    lineHeight: 32,
  },
  chipGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 18 },
  chip: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: Colors.hairline,
    backgroundColor: Colors.surface,
  },
  chipActive: { borderColor: Colors.primary, backgroundColor: Colors.primarySoft },
  chipText: { fontSize: 14, color: Colors.textPrimary, fontWeight: '700' },
  chipTextActive: { color: Colors.primaryDark, fontWeight: '900' },
  anyBtn: {
    alignSelf: 'stretch',
    paddingVertical: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.hairline,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
    backgroundColor: Colors.surface,
  },
  anyBtnActive: { borderColor: Colors.primary, backgroundColor: Colors.primarySoft },
  anyBtnText: { fontSize: 15, color: Colors.textSecondary, fontWeight: '800' },
  nextBtn: {
    backgroundColor: Colors.primary,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
  },
  nextBtnDisabled: { backgroundColor: Colors.surfaceAlt },
  nextBtnText: { color: '#fff', fontSize: 16, fontWeight: '900' },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 14,
  },
  greeting: {
    fontSize: 12,
    color: Colors.primaryDark,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  resultsTitle: { fontSize: 26, fontWeight: '900', color: Colors.textPrimary, marginTop: 2 },
  restartBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: Colors.primarySoft,
  },
  restartText: { color: Colors.primary, fontWeight: '800', fontSize: 14 },
  resultsList: { paddingHorizontal: 12, paddingBottom: 24 },
  resultRow: { justifyContent: 'space-between', marginBottom: 12 },
  resultCard: {
    flex: 1,
    marginHorizontal: 4,
    borderRadius: 18,
    backgroundColor: Colors.surface,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.hairline,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 16,
    elevation: 3,
  },
  resultImage: { width: '100%', aspectRatio: 1 },
  matchBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  matchBadgeText: { color: '#fff', fontSize: 11, fontWeight: '900' },
  resultInfo: { padding: 11 },
  resultName: { fontSize: 14, fontWeight: '900', color: Colors.textPrimary, marginBottom: 5, lineHeight: 18 },
  resultMeta: { fontSize: 12, color: Colors.textSecondary, fontWeight: '600' },
  specificNote: {
    margin: 16,
    padding: 14,
    borderRadius: 16,
    backgroundColor: Colors.primarySoft,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  specificNoteText: {
    flex: 1,
    fontSize: 13,
    color: Colors.primaryDark,
    fontWeight: '700',
    lineHeight: 18,
  },
});
