import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
} from "react-native";
import Svg, { Circle, Line, Text as SvgText } from "react-native-svg";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSequence,
  withTiming,
} from "react-native-reanimated";
import { Colors } from "@/constants/colors";
import { Suggestion } from "@/services/explorerApi";

type GraphCanvasProps = {
  center: string;
  suggestions: Suggestion[];
  previousNodes: string[];
  relaxed: boolean;
  onNodePress: (ingredient: string) => void;
};

const canvasWidth = Dimensions.get("window").width;
const viewportHeight = Dimensions.get("window").height * 0.55;
const TOP_PADDING = 60;
const BOTTOM_PADDING = 80;
const NODE_VERTICAL_SPACING = 125;
const SATELLITES_AREA_HEIGHT = 280;
const SATELLITE_RADIUS = 180;
const HORIZONTAL_MARGIN = 4;
const SATELLITE_ARC_START = 200;
const SATELLITE_ARC_DEGREES = 140;
const CURRENT_NODE_RADIUS = 56;
const CHAIN_NODE_RADIUS = 42;
const SATELLITE_NODE_RADIUS_FALLBACK = 56;
const SATELLITE_NODE_MIN_OPACITY = 0.5;
const HIT_SIZE = 104;

function labelLines(label: string, maxChars = 10, maxLines = 3): string[] {
  const words = label.split(" ").filter(Boolean);
  if (!words.length) return [""];
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars) {
      current = next;
      continue;
    }
    if (current) lines.push(current);
    current = word;
  }
  if (current) lines.push(current);

  const trimmed = lines.slice(0, maxLines);
  if (lines.length > maxLines) {
    trimmed[maxLines - 1] =
      `${trimmed[maxLines - 1].slice(0, Math.max(maxChars - 1, 1))}...`;
  }
  return trimmed.map((line) =>
    line.length > maxChars
      ? `${line.slice(0, Math.max(maxChars - 1, 1))}...`
      : line,
  );
}

function nodeFontSize(label: string, baseSize: number): number {
  if (label.length > 24) return Math.max(baseSize - 3, 8);
  if (label.length > 16) return Math.max(baseSize - 2, 9);
  if (label.length > 11) return Math.max(baseSize - 1, 10);
  return baseSize;
}

function satelliteAngle(index: number, count: number): number {
  if (count <= 1) return 270;
  return SATELLITE_ARC_START + index * (SATELLITE_ARC_DEGREES / (count - 1));
}

function suggestionWeights(suggestions: Suggestion[]): Record<string, number> {
  const scores = suggestions.map((item) => item.score ?? 0);
  const min = Math.min(...scores, 0);
  const max = Math.max(...scores, 0);
  if (max <= min) {
    return Object.fromEntries(
      suggestions.map((item) => [item.ingredient, 0.5]),
    );
  }
  return Object.fromEntries(
    suggestions.map((item) => [
      item.ingredient,
      ((item.score ?? 0) - min) / (max - min),
    ]),
  );
}

function satelliteVisuals(item: Suggestion, weights: Record<string, number>) {
  const weight = weights[item.ingredient] ?? 0.5;
  return {
    radius: 40,
    opacity:
      SATELLITE_NODE_MIN_OPACITY + weight * (1 - SATELLITE_NODE_MIN_OPACITY),
  };
}

export function GraphCanvas({
  center,
  suggestions,
  previousNodes,
  relaxed,
  onNodePress,
}: GraphCanvasProps) {
  const scrollRef = useRef<ScrollView>(null);
  const [selectedSatellite, setSelectedSatellite] = useState<string | null>(
    null,
  );
  const satelliteOpacity = useSharedValue(0);
  const pulseScale = useSharedValue(1);
  const chainNodes = useMemo(
    () => [...previousNodes, center].filter(Boolean),
    [center, previousNodes],
  );
  const canvasHeight =
    TOP_PADDING +
    Math.max(chainNodes.length, 1) * NODE_VERTICAL_SPACING +
    SATELLITES_AREA_HEIGHT +
    BOTTOM_PADDING;
  const centerX = canvasWidth / 2;
  const currentY =
    TOP_PADDING + Math.max(chainNodes.length - 1, 0) * NODE_VERTICAL_SPACING;
  const satelliteRadius = Math.min(
    SATELLITE_RADIUS,
    canvasWidth / 2 - SATELLITE_NODE_RADIUS_FALLBACK - HORIZONTAL_MARGIN,
  );
  const weights = useMemo(() => suggestionWeights(suggestions), [suggestions]);

  useEffect(() => {
    satelliteOpacity.value = 0;
    satelliteOpacity.value = withTiming(1, { duration: 200 });
    setSelectedSatellite(null);
    requestAnimationFrame(() => {
      scrollRef.current?.scrollToEnd({ animated: true });
    });
  }, [center, satelliteOpacity, suggestions]);

  const satelliteFadeStyle = useAnimatedStyle(() => ({
    opacity: satelliteOpacity.value,
  }));

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const handlePress = (ingredient: string) => {
    setSelectedSatellite(ingredient);
    satelliteOpacity.value = withTiming(0, { duration: 200 });
    pulseScale.value = withSequence(
      withTiming(1.15, { duration: 150 }),
      withTiming(1, { duration: 150 }),
    );
    setTimeout(() => onNodePress(ingredient), 220);
  };

  return (
    <View style={styles.viewport}>
      <ScrollView
        ref={scrollRef}
        showsVerticalScrollIndicator={false}
        scrollEnabled
        style={styles.scroll}
      >
        <View style={[styles.canvas, { height: canvasHeight }]}>
          <Svg width={canvasWidth} height={canvasHeight}>
            {chainNodes.slice(0, -1).map((_, index) => {
              const y1 = TOP_PADDING + index * NODE_VERTICAL_SPACING;
              const y2 = TOP_PADDING + (index + 1) * NODE_VERTICAL_SPACING;
              return (
                <Line
                  key={`chain-line-${index}`}
                  x1={centerX}
                  y1={y1}
                  x2={centerX}
                  y2={y2}
                  stroke={Colors.border}
                  strokeWidth={1.5}
                />
              );
            })}

            {chainNodes.map((node, index) => {
              const isCurrent = index === chainNodes.length - 1;
              const y = TOP_PADDING + index * NODE_VERTICAL_SPACING;
              return (
                <React.Fragment key={`${node}-${index}`}>
                  {isCurrent && relaxed ? (
                    <Circle
                      cx={centerX}
                      cy={y}
                      r={CURRENT_NODE_RADIUS + 8}
                      fill="none"
                      stroke={Colors.warning}
                      strokeWidth={1.5}
                      strokeDasharray="5 4"
                    />
                  ) : null}
                  <Circle
                    cx={centerX}
                    cy={y}
                    r={isCurrent ? CURRENT_NODE_RADIUS : CHAIN_NODE_RADIUS}
                    fill={isCurrent ? Colors.primary : Colors.textTertiary}
                  />
                  {labelLines(node, isCurrent ? 15 : 12).map(
                    (line, lineIndex, lines) => {
                      const fontSize = nodeFontSize(node, isCurrent ? 13 : 11);
                      const lineHeight = fontSize + 2;
                      return (
                        <SvgText
                          key={`${node}-${line}`}
                          x={centerX}
                          y={
                            y +
                            (lineIndex - (lines.length - 1) / 2) * lineHeight +
                            fontSize / 3
                          }
                          fill={Colors.textInverse}
                          fontSize={fontSize}
                          fontWeight="800"
                          textAnchor="middle"
                        >
                          {line}
                        </SvgText>
                      );
                    },
                  )}
                </React.Fragment>
              );
            })}
          </Svg>

          <Animated.View
            pointerEvents="none"
            style={[StyleSheet.absoluteFill, satelliteFadeStyle]}
          >
            <Svg width={canvasWidth} height={canvasHeight}>
              {suggestions.map((item, index) => {
                const angle =
                  satelliteAngle(index, suggestions.length) * (Math.PI / 180);
                const x = centerX + satelliteRadius * Math.cos(angle);
                const y = currentY - satelliteRadius * Math.sin(angle);
                const visual = satelliteVisuals(item, weights);
                return (
                  <React.Fragment key={`satellite-${item.ingredient}`}>
                    <Line
                      x1={centerX}
                      y1={currentY}
                      x2={x}
                      y2={y}
                      stroke={Colors.border}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                    />
                    <Circle
                      cx={x}
                      cy={y}
                      r={visual.radius}
                      fill={Colors.accent}
                      opacity={visual.opacity}
                    />
                    {labelLines(item.ingredient, 12).map(
                      (line, lineIndex, lines) => {
                        const fontSize = nodeFontSize(item.ingredient, 11);
                        const lineHeight = fontSize + 2;
                        return (
                          <SvgText
                            key={`${item.ingredient}-${line}`}
                            x={x}
                            y={
                              y +
                              (lineIndex - (lines.length - 1) / 2) *
                                lineHeight +
                              fontSize / 3
                            }
                            fill={Colors.textInverse}
                            fontSize={fontSize}
                            fontWeight="700"
                            textAnchor="middle"
                          >
                            {line}
                          </SvgText>
                        );
                      },
                    )}
                  </React.Fragment>
                );
              })}
            </Svg>
          </Animated.View>

          {selectedSatellite
            ? suggestions.map((item, index) => {
                if (item.ingredient !== selectedSatellite) return null;
                const angle =
                  satelliteAngle(index, suggestions.length) * (Math.PI / 180);
                const x = centerX + satelliteRadius * Math.cos(angle);
                const y = currentY - satelliteRadius * Math.sin(angle);
                const visual = satelliteVisuals(item, weights);
                return (
                  <React.Fragment key={`selected-${item.ingredient}`}>
                    <Svg
                      pointerEvents="none"
                      width={canvasWidth}
                      height={canvasHeight}
                      style={StyleSheet.absoluteFill}
                    >
                      <Line
                        x1={centerX}
                        y1={currentY}
                        x2={x}
                        y2={y}
                        stroke={Colors.border}
                        strokeWidth={1.5}
                        strokeDasharray="4 3"
                      />
                    </Svg>
                    <Animated.View
                      pointerEvents="none"
                      style={[
                        styles.selectedNode,
                        { left: x - HIT_SIZE / 2, top: y - HIT_SIZE / 2 },
                        pulseStyle,
                      ]}
                    >
                      <Svg width={HIT_SIZE} height={HIT_SIZE}>
                        <Circle
                          cx={HIT_SIZE / 2}
                          cy={HIT_SIZE / 2}
                          r={visual.radius}
                          fill={Colors.accent}
                          opacity={visual.opacity}
                        />
                        {labelLines(item.ingredient, 12).map(
                          (line, lineIndex, lines) => {
                            const fontSize = nodeFontSize(item.ingredient, 11);
                            const lineHeight = fontSize + 2;
                            return (
                              <SvgText
                                key={`${item.ingredient}-selected-${line}`}
                                x={HIT_SIZE / 2}
                                y={
                                  HIT_SIZE / 2 +
                                  (lineIndex - (lines.length - 1) / 2) *
                                    lineHeight +
                                  fontSize / 3
                                }
                                fill={Colors.textInverse}
                                fontSize={fontSize}
                                fontWeight="700"
                                textAnchor="middle"
                              >
                                {line}
                              </SvgText>
                            );
                          },
                        )}
                      </Svg>
                    </Animated.View>
                  </React.Fragment>
                );
              })
            : null}

          {suggestions.map((item, index) => {
            const angle =
              satelliteAngle(index, suggestions.length) * (Math.PI / 180);
            const x =
              centerX + satelliteRadius * Math.cos(angle) - HIT_SIZE / 2;
            const y =
              currentY - satelliteRadius * Math.sin(angle) - HIT_SIZE / 2;
            return (
              <Animated.View
                key={`hit-${item.ingredient}`}
                style={[
                  styles.hitWrap,
                  { left: x, top: y },
                  selectedSatellite === item.ingredient && pulseStyle,
                ]}
              >
                <Pressable
                  style={styles.hit}
                  disabled={selectedSatellite !== null}
                  onPress={() => handlePress(item.ingredient)}
                />
              </Animated.View>
            );
          })}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  viewport: {
    width: canvasWidth,
    height: viewportHeight,
  },
  scroll: {
    flex: 1,
  },
  canvas: {
    width: canvasWidth,
    position: "relative",
  },
  hitWrap: {
    position: "absolute",
    width: HIT_SIZE,
    height: HIT_SIZE,
    borderRadius: HIT_SIZE / 2,
  },
  selectedNode: {
    position: "absolute",
    width: HIT_SIZE,
    height: HIT_SIZE,
    borderRadius: HIT_SIZE / 2,
  },
  hit: {
    width: HIT_SIZE,
    height: HIT_SIZE,
    borderRadius: HIT_SIZE / 2,
  },
});
