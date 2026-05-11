export const Colors = {
  primary: '#E15F2A',
  primaryLight: '#FFE2D2',
  primaryDark: '#9A3412',
  primarySoft: '#FFF3EC',

  background: '#F7F3ED',
  surface: '#FFFFFF',
  surfaceAlt: '#EFE8DC',
  surfaceMuted: '#FAF7F2',
  surfaceElevated: '#FFFCF8',

  textPrimary: '#231F1A',
  textSecondary: '#6F665C',
  textTertiary: '#A69B8F',
  textInverse: '#FFFFFF',

  border: '#E4D9CC',
  borderFocus: '#E15F2A',
  hairline: '#EFE7DD',

  success: '#2E7D55',
  successSoft: '#DDEFE5',
  error: '#C24135',
  errorSoft: '#F7DED9',
  warning: '#B7791F',
  warningSoft: '#F8E8BF',
  accent: '#2F6F73',
  accentSoft: '#DDECEC',
  plum: '#6B4E71',
  plumSoft: '#E9DFEA',

  tabActive: '#231F1A',
  tabInactive: '#9A9187',
  tabBar: '#FFFCF8',
  shadow: '#2A1A10',
} as const;

export type ColorKey = keyof typeof Colors;
