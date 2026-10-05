export type UnknownRecord = Record<string, unknown>;

export function asRecord(value: unknown): UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as UnknownRecord : {};
}

export function displayText(value: unknown, fallback = '—') {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

export function displayNumber(value: unknown, suffix = '', digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : '—';
}

export function displayPercent(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

export function warningTitle(value: unknown) {
  return ({
    RED: 'EXTREME RAINFALL SIGNAL',
    ORANGE: 'HEAVY RAINFALL SIGNAL',
    YELLOW: 'MODERATE RAINFALL SIGNAL',
    GREEN: 'LOW · NO HEAVY-RAIN RULE TRIGGERED',
    UNAVAILABLE: 'FORECAST UNAVAILABLE',
  } as Record<string, string>)[displayText(value, 'UNAVAILABLE')] || 'FORECAST STATUS UNAVAILABLE';
}

export function warningCategory(value: unknown) {
  return ({ RED: 'EXTREME', ORANGE: 'HEAVY', YELLOW: 'MODERATE', GREEN: 'LOW' } as Record<string, string>)[displayText(value, '')] || 'UNAVAILABLE';
}

export function warningTone(value: unknown) {
  return ({
    RED: 'border-[#D64545]/35 bg-[#FCEBEC] text-[#8e2525]',
    ORANGE: 'border-[#d98854]/35 bg-[#fff1e8] text-[#9a4b1f]',
    YELLOW: 'border-[#E97BA8]/40 bg-[#fff1f6] text-[#87425f]',
    GREEN: 'border-[#2E8B57]/30 bg-[#EAF5EE] text-[#246b43]',
  } as Record<string, string>)[displayText(value, '')] || 'border-[#d9e2e8] bg-[#f1f4f6] text-[#536a7a]';
}

export function operationalMode(value: unknown) {
  return value === 'LIVE' ? 'LIVE' : value === 'DEMO' ? 'DEMO' : value === 'REFERENCE' ? 'REFERENCE' : 'UNAVAILABLE';
}

export function utcLabel(value: unknown) {
  if (typeof value !== 'string' || !value) return '—';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : `${parsed.toISOString().replace('T', ' ').slice(0, 16)} UTC`;
}
