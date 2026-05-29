import { useState, useCallback } from 'react';
import {
  recommendationApi,
  RecQuestion,
  RecScoredRecipe,
  RecProgress,
  DietaryProfile,
  ApiError,
} from '@/services/api';

export type RecPhase = 'idle' | 'questioning' | 'done' | 'error';

interface RecState {
  phase: RecPhase;
  sessionId: string | null;
  currentQuestion: RecQuestion | null;
  progress: RecProgress;
  entropy: number | null;
  questionsAsked: number;
  results: RecScoredRecipe[];
  resultsCount: number;
  loading: boolean;
  error: string | null;
}

const INITIAL: RecState = {
  phase: 'idle',
  sessionId: null,
  currentQuestion: null,
  progress: { current: 1, max: 15 },
  entropy: null,
  questionsAsked: 0,
  results: [],
  resultsCount: 0,
  loading: false,
  error: null,
};

export function useRecommendation() {
  const [state, setState] = useState<RecState>(INITIAL);

  const startSession = useCallback(async (dietary?: DietaryProfile) => {
    setState({ ...INITIAL, loading: true });
    try {
      const res = await recommendationApi.start(dietary);
      setState({
        ...INITIAL,
        phase: 'questioning',
        sessionId: res.session_id,
        currentQuestion: res.question,
        progress: res.progress,
        loading: false,
      });
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Could not start session';
      setState({ ...INITIAL, phase: 'error', error: msg });
    }
  }, []);

  const submitAnswer = useCallback(
    async (questionId: string, answer: string | string[]) => {
      const { sessionId } = state;
      if (!sessionId) return;

      setState(s => ({ ...s, loading: true }));
      try {
        const res = await recommendationApi.answer(sessionId, questionId, answer);

        if (res.status === 'done') {
          setState(s => ({
            ...s,
            phase: 'done',
            results: res.results ?? [],
            resultsCount: res.results_count ?? (res.results?.length ?? 0),
            currentQuestion: null,
            questionsAsked: res.questions_asked ?? s.questionsAsked,
            entropy: res.entropy,
            loading: false,
          }));
        } else {
          setState(s => ({
            ...s,
            currentQuestion: res.question,
            progress: res.progress ?? s.progress,
            entropy: res.entropy,
            questionsAsked: res.questions_asked ?? s.questionsAsked,
            loading: false,
          }));
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : 'Could not submit answer';
        setState(s => ({ ...s, loading: false, error: msg }));
      }
    },
    [state],
  );

  const reset = useCallback(() => {
    setState(INITIAL);
  }, []);

  return { ...state, startSession, submitAnswer, reset };
}
