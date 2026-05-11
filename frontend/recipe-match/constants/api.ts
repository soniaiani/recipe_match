import { Platform } from 'react-native';
import Constants from 'expo-constants';

function getDevHost(): string {
  if (Platform.OS === 'web') return 'localhost';

  const hostUri = Constants.expoConfig?.hostUri;
  if (hostUri) {
    return hostUri.split(':')[0];
  }

  if (Platform.OS === 'android') return '10.0.2.2';
  return 'localhost';
}

export const API_BASE_URL = __DEV__
  ? `http://${getDevHost()}:8000`
  : 'https://your-production-api.com';
