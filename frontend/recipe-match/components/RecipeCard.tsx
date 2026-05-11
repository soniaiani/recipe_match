import React from 'react';
import {
  View, Text, StyleSheet, Pressable, Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/colors';
import { RecipeSummary } from '@/services/api';

interface Props {
  recipe: RecipeSummary;
  score?: number;
  onPress: () => void;
  onSave?: () => void;
  isSaved?: boolean;
}

const CUISINE_LABELS: Record<string, string> = {
  italian: 'Italian', asian: 'Asian', american: 'American',
  mexican: 'Mexican', french: 'French', mediterranean: 'Mediterranean',
  indian: 'Indian', other: 'Other',
};

export function RecipeCard({ recipe, score, onPress, onSave, isSaved }: Props) {
  const badges: string[] = [];
  if (recipe.is_quick) badges.push('Quick');
  if (recipe.is_vegan) badges.push('Vegan');
  else if (recipe.is_vegetarian) badges.push('Vegetarian');
  if (recipe.is_gluten_free) badges.push('GF');

  return (
    <Pressable style={styles.card} onPress={onPress}>
      <View style={styles.imageWrap}>
        {recipe.image_url ? (
          <Image source={{ uri: recipe.image_url }} style={styles.image} />
        ) : (
          <View style={styles.imagePlaceholder}>
            <Ionicons name="restaurant-outline" size={34} color={Colors.textTertiary} />
          </View>
        )}
        <View style={styles.imageScrim} />

        {score != null && (
          <View style={styles.scorePill}>
            <Ionicons name="sparkles" size={12} color={Colors.textInverse} />
            <Text style={styles.scoreText}>{Math.round(score)}%</Text>
          </View>
        )}

        {onSave && (
          <Pressable onPress={onSave} style={styles.saveBtn} hitSlop={10}>
            <Ionicons
              name={isSaved ? 'heart' : 'heart-outline'}
              size={20}
              color={isSaved ? Colors.primary : Colors.textPrimary}
            />
          </Pressable>
        )}
      </View>

      <View style={styles.body}>
        <Text style={styles.name} numberOfLines={2}>{recipe.name}</Text>

        <View style={styles.meta}>
          {recipe.cuisine && (
            <View style={styles.metaItem}>
              <Ionicons name="earth-outline" size={13} color={Colors.textSecondary} />
              <Text style={styles.metaText}>
                {CUISINE_LABELS[recipe.cuisine] ?? recipe.cuisine}
              </Text>
            </View>
          )}
          {recipe.total_minutes != null && (
            <View style={styles.metaItem}>
              <Ionicons name="time-outline" size={13} color={Colors.textSecondary} />
              <Text style={styles.metaText}>{recipe.total_minutes} min</Text>
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
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 18,
    marginHorizontal: 16,
    marginBottom: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.hairline,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 4,
  },
  imageWrap: {
    height: 188,
    backgroundColor: Colors.surfaceAlt,
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  imageScrim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(35, 31, 26, 0.10)',
  },
  imagePlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: Colors.surfaceMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scorePill: {
    position: 'absolute',
    left: 12,
    top: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(35, 31, 26, 0.78)',
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  scoreText: {
    color: Colors.textInverse,
    fontSize: 12,
    fontWeight: '800',
  },
  saveBtn: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: 'rgba(255, 252, 248, 0.92)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  body: {
    padding: 15,
  },
  name: {
    fontSize: 17,
    fontWeight: '800',
    color: Colors.textPrimary,
    lineHeight: 23,
  },
  meta: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 9,
    flexWrap: 'wrap',
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    fontSize: 13,
    color: Colors.textSecondary,
    fontWeight: '600',
  },
  badges: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 10,
    flexWrap: 'wrap',
  },
  badge: {
    backgroundColor: Colors.primarySoft,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: Colors.primaryLight,
  },
  badgeText: {
    fontSize: 11,
    color: Colors.primaryDark,
    fontWeight: '700',
  },
});
