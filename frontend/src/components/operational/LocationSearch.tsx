'use client';

import { useEffect, useMemo, useState } from 'react';
import type { DistrictFeature } from '@/components/district/DistrictBoundaryMap';
import type { LocationSelection } from './OperationalWorkspace';

const GEOCODING_URL = 'https://geocoding-api.open-meteo.com/v1/search';
const CACHE_KEY = 'anvaya:open-meteo-geocoding:v1';
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

type GeocodingResult = {
  id?: number;
  name: string;
  latitude: number;
  longitude: number;
  country?: string;
  country_code?: string;
  admin1?: string;
  admin2?: string;
  feature_code?: string;
};
type CacheEntry = { savedAt: number; results: GeocodingResult[] };

function readCache(): Record<string, CacheEntry> {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(CACHE_KEY) || '{}');
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, CacheEntry> : {};
  } catch { return {}; }
}

function fromCache(query: string) {
  const entry = readCache()[query.toLocaleLowerCase()];
  return entry && Date.now() - entry.savedAt < CACHE_TTL_MS ? entry.results : null;
}

function saveCache(query: string, results: GeocodingResult[]) {
  try {
    const cache = readCache();
    cache[query.toLocaleLowerCase()] = { savedAt: Date.now(), results };
    const recent = Object.entries(cache).sort((a, b) => b[1].savedAt - a[1].savedAt).slice(0, 20);
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(Object.fromEntries(recent)));
  } catch { /* Search remains usable if storage is blocked or full. */ }
}

function indiaFirst(results: GeocodingResult[], query: string) {
  const explicitCountry = query.includes(',');
  return [...results].sort((a, b) => {
    if (!explicitCountry) {
      const indiaOrder = Number(b.country_code?.toUpperCase() === 'IN') - Number(a.country_code?.toUpperCase() === 'IN');
      if (indiaOrder) return indiaOrder;
    }
    return (b.feature_code?.startsWith('PPL') ? 1 : 0) - (a.feature_code?.startsWith('PPL') ? 1 : 0);
  });
}

function placeCaption(place: GeocodingResult) {
  return [place.admin2, place.admin1, place.country].filter(Boolean).join(' · ');
}

export default function LocationSearch({
  districts, selectedLocation, onSelect, label = 'Search city, village, district or locality', compact = false,
}: {
  districts: DistrictFeature[];
  selectedLocation?: LocationSelection | null;
  onSelect: (location: LocationSelection) => void;
  label?: string;
  compact?: boolean;
}) {
  const [query, setQuery] = useState('');
  const [places, setPlaces] = useState<GeocodingResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [geocodingError, setGeocodingError] = useState(false);
  const matches = useMemo(() => {
    const term = query.trim().toLocaleLowerCase();
    if (!term) return [];
    return districts.filter((feature) => `${String(feature.properties?.district ?? '')} ${String(feature.properties?.state_name ?? '')}`.toLocaleLowerCase().includes(term)).slice(0, 6);
  }, [districts, query]);

  useEffect(() => {
    const term = query.trim();
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      if (term.length < 2) {
        setPlaces([]); setSearching(false); setGeocodingError(false);
        return;
      }
      setGeocodingError(false);
      const cached = fromCache(term);
      if (cached) { setPlaces(indiaFirst(cached, term)); setSearching(false); }
      else { setPlaces([]); setSearching(true); }
      try {
        const url = new URL(GEOCODING_URL);
        url.searchParams.set('name', term);
        url.searchParams.set('count', '15');
        url.searchParams.set('language', 'en');
        const response = await fetch(url, { signal: controller.signal, headers: { Accept: 'application/json' } });
        if (!response.ok) throw new Error(`Geocoding returned HTTP ${response.status}`);
        const payload: unknown = await response.json();
        const rows = typeof payload === 'object' && payload !== null && Array.isArray((payload as { results?: unknown }).results)
          ? (payload as { results: GeocodingResult[] }).results.filter((place) => Number.isFinite(place.latitude) && Number.isFinite(place.longitude)) : [];
        saveCache(term, rows);
        if (active) { setPlaces(indiaFirst(rows, term)); setGeocodingError(false); }
      } catch (error) {
        if (active && !(error instanceof DOMException && error.name === 'AbortError')) {
          setGeocodingError(true);
          if (!cached) setPlaces([]);
        }
      } finally { if (active) setSearching(false); }
    }, term.length < 2 ? 0 : 350);
    return () => { active = false; window.clearTimeout(timer); controller.abort(); };
  }, [query]);

  const selectedText = selectedLocation ? `${selectedLocation.name}${selectedLocation.state ? ` · ${selectedLocation.state}` : ''}` : '';
  const hasResults = places.length > 0 || matches.length > 0;

  return <div className="location-search min-w-0">
    <label htmlFor={compact ? 'map-location-search' : 'home-location-search'} className="mb-1.5 block text-[11px] font-semibold text-[#12324A]">{label}</label>
    <div className="relative">
      <input
        id={compact ? 'map-location-search' : 'home-location-search'}
        type="search" value={query} onChange={(event) => setQuery(event.target.value)}
        placeholder="Search city, town, village or district" autoComplete="off" aria-describedby="location-search-help"
        aria-controls="location-search-results"
        className="h-11 w-full rounded-md border border-[#c6d3dd] bg-white px-3 text-sm text-[#12324A] outline-none placeholder:text-[#7b8b98] focus:border-[#1769AA] focus:ring-2 focus:ring-[#1769AA]/15"
      />
      {hasResults && <div id="location-search-results" className="absolute left-0 right-0 top-full z-30 mt-1 max-h-80 overflow-y-auto rounded-md border border-[#d5dfe6] bg-white shadow-lg" role="listbox" aria-label="Location search results">
        {places.map((place, index) => <button key={`place:${place.id ?? `${place.latitude}:${place.longitude}`}`} type="button" role="option" aria-selected={false}
          onClick={() => {
            const address = placeCaption(place);
            onSelect({
              key: `place:${place.id ?? `${place.latitude.toFixed(5)},${place.longitude.toFixed(5)}`}`,
              kind: 'place', name: place.name, address, country: place.country, admin1: place.admin1, admin2: place.admin2,
              state: place.admin1, latitude: place.latitude, longitude: place.longitude,
            });
            setQuery(''); setPlaces([]);
          }}
          className="flex w-full items-start justify-between gap-3 border-b border-[#e7edf1] px-3 py-2.5 text-left last:border-b-0 hover:bg-[#f1f7fb] focus:bg-[#f1f7fb] focus:outline-none">
          <span><span className="block text-sm font-medium text-[#12324A]">{place.name}</span><span className="mt-0.5 block text-xs text-[#667b8a]">{placeCaption(place) || place.country_code}</span></span>
          {!index && place.country_code?.toUpperCase() === 'IN' && <span className="shrink-0 text-[9px] font-semibold text-[#1769AA]">INDIA</span>}
        </button>)}
        {matches.map((feature) => {
          const properties = feature.properties || {};
          const id = String(properties.anvaya_district_id ?? '');
          return <button key={`district:${id}`} type="button" role="option" aria-selected={selectedLocation?.kind === 'district' && selectedLocation.districtId === id}
            onClick={() => { onSelect({ key: `district:${id}`, kind: 'district', districtId: id, name: String(properties.district ?? 'District'), state: String(properties.state_name ?? '') || undefined, country: 'India' }); setQuery(''); setPlaces([]); }}
            className="flex w-full items-center justify-between gap-3 border-b border-[#e7edf1] px-3 py-2.5 text-left last:border-b-0 hover:bg-[#f1f7fb] focus:bg-[#f1f7fb] focus:outline-none">
            <span className="text-sm font-medium text-[#12324A]">{String(properties.district ?? 'District')}</span><span className="text-xs text-[#667b8a]">{String(properties.state_name ?? '')} · India</span>
          </button>;
        })}
      </div>}
    </div>
    <p id="location-search-help" className="mt-1.5 text-[11px] text-[#667b8a]">
      {searching ? 'Searching Open-Meteo and the local district directory…' : geocodingError ? 'External location search is unavailable; matching districts remain searchable below.' : 'Open-Meteo location results are cached locally for 24 hours. District search remains available offline.'}
    </p>
    {geocodingError && !matches.length && <p role="status" className="mt-1 text-xs text-[#8a5a1f]">Geocoding is temporarily unavailable; try a district or state name if it is in the local directory.</p>}
    {selectedText && <p className="mt-2 truncate text-xs font-medium text-[#12324A]" title={selectedText}>Selected · {selectedText}</p>}
    {query.trim().length > 1 && !searching && !hasResults && <p role="status" className="mt-2 text-xs text-[#667b8a]">No results. Try another spelling or choose a district boundary on the map.</p>}
  </div>;
}
