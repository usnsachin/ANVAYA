// Local development uses the same-origin Next.js proxy. A deployed frontend
// receives the public FastAPI origin at build time and calls its /api routes.
const configuredApiOrigin = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, '');
const API_URL = configuredApiOrigin
  ? (configuredApiOrigin.endsWith('/api') ? configuredApiOrigin : `${configuredApiOrigin}/api`)
  : '/api';

async function getJson(path: string) {
  const response = await fetch(`${API_URL}${path}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`API ${response.status}: ${response.statusText}`);
  return response.json();
}

export async function fetchHealth() {
  const res = await fetch(`${API_URL}/health`, { cache: 'no-store' });
  return res.json();
}

export async function fetchForecasts() {
  const res = await fetch(`${API_URL}/forecast`, { cache: 'no-store' });
  return res.json();
}

export async function fetchDistricts() {
  const res = await fetch(`${API_URL}/districts`, { cache: 'no-store' });
  return res.json();
}

export async function fetchVerification() {
  const res = await fetch(`${API_URL}/verification`, { cache: 'no-store' });
  return res.json();
}

export async function fetchRegimes() {
  const res = await fetch(`${API_URL}/regimes`, { cache: 'no-store' });
  return res.json();
}

export async function fetchOperationalForecast(latitude: number, longitude: number) {
  const params = new URLSearchParams({ latitude: String(latitude), longitude: String(longitude) });
  return getJson(`/operational/forecast?${params.toString()}`);
}

export async function fetchOperationalDistricts(search = '') {
  const params = new URLSearchParams({ limit: '200' });
  if (search.trim()) params.set('search', search.trim());
  return getJson(`/operational/districts?${params.toString()}`);
}

export async function fetchOperationalDistrict(districtId: string) {
  return getJson(`/operational/districts/${encodeURIComponent(districtId)}`);
}

export async function fetchOperationalMap(districtId?: string, layer = 'district_warning') {
  const params = new URLSearchParams({ layer });
  if (districtId) params.set('district_id', districtId);
  return getJson(`/operational/map?${params.toString()}`);
}
