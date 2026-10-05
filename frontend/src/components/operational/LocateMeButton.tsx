'use client';

import { useState } from 'react';
import { useOperationalWorkspace } from './OperationalWorkspace';
import type { DistrictFeature } from '@/components/district/DistrictBoundaryMap';

function positions(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  if (value.length >= 2 && typeof value[0] === 'number' && typeof value[1] === 'number') return [value as number[]];
  return value.flatMap(positions);
}

function nearestDistrict(features: DistrictFeature[], latitude: number, longitude: number) {
  let nearest: { feature: DistrictFeature; distance: number } | null = null;
  for (const feature of features) {
    const points = positions(feature.geometry.coordinates);
    if (!points.length) continue;
    const center = points.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0]);
    const dx = center[0] / points.length - longitude;
    const dy = center[1] / points.length - latitude;
    const distance = dx * dx + dy * dy;
    if (!nearest || distance < nearest.distance) nearest = { feature, distance };
  }
  return nearest?.feature;
}

export default function LocateMeButton() {
  const { boundaries, selectLocation, requestLocationForecast, requestDistrictForecast } = useOperationalWorkspace();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const locate = () => {
    if (!navigator.geolocation) {
      setMessage('Location is not supported by this browser.');
      return;
    }
    setLoading(true);
    setMessage('Waiting for browser location permission…');
    navigator.geolocation.getCurrentPosition(async ({ coords }) => {
      const { latitude, longitude } = coords;
      if (latitude < 8 || latitude > 37 || longitude < 68 || longitude > 97) {
        setMessage('Location found, but it is outside the supported India forecast domain.');
        setLoading(false);
        return;
      }
      const nearbyFeature = nearestDistrict(boundaries?.features ?? [], latitude, longitude);
      const place = {
        key: `geolocation:${latitude.toFixed(5)},${longitude.toFixed(5)}`,
        kind: 'place' as const,
        name: nearbyFeature
          ? `Near ${String(nearbyFeature.properties?.district ?? 'current location')}`
          : 'Current location',
        state: String(nearbyFeature?.properties?.state_name ?? '') || undefined,
        latitude,
        longitude,
      };
      selectLocation(place);
      setMessage('Location found. Checking ANVAYA forecast availability…');
      const forecast = await requestLocationForecast(place);
      if (forecast && forecast.anvaya_corrected != null) {
        setMessage(`Location found · coordinate-based ANVAYA forecast${coords.accuracy ? ` · ±${Math.round(coords.accuracy)} m GPS accuracy` : ''}.`);
        setLoading(false);
        return;
      }
      const feature = nearbyFeature;
      const districtId = String(feature?.properties?.anvaya_district_id ?? '');
      if (feature && districtId) {
        const district = {
          key: `district:${districtId}`,
          kind: 'district' as const,
          districtId,
          name: String(feature.properties?.district ?? 'Nearest supported district'),
          state: String(feature.properties?.state_name ?? '') || undefined,
        };
        selectLocation(district);
        const districtForecast = await requestDistrictForecast(districtId);
        setMessage(districtForecast?.status === 'LIVE'
          ? 'Location found · forecast available for the nearest supported district, not an exact locality forecast.'
          : 'Location found · nearest supported district selected, but its forecast is unavailable.');
      } else {
        setMessage('Location found, but no supported ANVAYA forecast location is available nearby.');
      }
      setLoading(false);
    }, (error) => {
      setMessage(error.code === error.PERMISSION_DENIED
        ? 'Location permission was not granted. Search for a place or district instead.'
        : 'Your current location could not be resolved. Search for a place or district instead.');
      setLoading(false);
    }, { enableHighAccuracy: false, timeout: 12000, maximumAge: 60000 });
  };

  return <div className="min-w-0">
    <button type="button" onClick={locate} disabled={loading} className="min-h-10 whitespace-nowrap rounded-md border border-[#b7c9d5] bg-white px-3 text-xs font-semibold text-[#1769AA] hover:bg-[#f1f7fb] disabled:opacity-55">
      {loading ? 'LOCATING…' : 'LOCATE ME'}
    </button>
    {message && <p role="status" className="mt-1.5 text-[11px] leading-relaxed text-[#536a7a]">{message}</p>}
  </div>;
}
