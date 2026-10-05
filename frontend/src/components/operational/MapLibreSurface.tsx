'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { AttributionControl, Map as MapLibreMap, Marker, NavigationControl, Popup, setWorkerUrl, type GeoJSONSource, type Marker as MapMarker } from 'maplibre-gl';
import type { ReactNode } from 'react';
import type { FeatureCollection } from '@/components/district/DistrictBoundaryMap';
import type { MapStyle, OperationalLayer } from './OperationalWorkspace';

const STYLE_URL: Record<Exclude<MapStyle, 'satellite'>, string> = {
  standard: 'https://tiles.openfreemap.org/styles/bright',
  light: 'https://tiles.openfreemap.org/styles/positron',
  dark: 'https://tiles.openfreemap.org/styles/dark',
};

setWorkerUrl('/maplibre/maplibre-gl-worker.mjs');

function featureColor(properties: Record<string, unknown>, layer: OperationalLayer) {
  const neutral = '#e4ebef';
  if (properties.forecast_status !== 'LIVE' || properties.forecast_value == null) return neutral;
  const warning = String(properties.warning_level ?? '');
  if (layer === 'risk') return ({ GREEN: '#2E8B57', YELLOW: '#E9B949', ORANGE: '#D64545', RED: '#D34F9A' } as Record<string, string>)[warning] || neutral;
  if (layer === 'regime') {
    const regime = String(properties.regime ?? '').toLowerCase();
    return regime.includes('active') || regime.includes('monsoon') ? '#2E8B57' : regime.includes('break') ? '#9270A8' : '#68859A';
  }
  const value = properties.forecast_value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return neutral;
  if (layer === 'correction') return value > 0 ? '#D64545' : value < 0 ? '#1769AA' : '#79909F';
  return value >= 204.5 ? '#D34F9A' : value >= 115.6 ? '#D64545' : value >= 64.5 ? '#E9B949' : '#2E8B57';
}

function mapData(collection: FeatureCollection | null, layer: OperationalLayer) {
  if (!collection) return { type: 'FeatureCollection' as const, features: [] };
  return {
    type: 'FeatureCollection' as const,
    features: collection.features.map((feature) => ({
      ...feature,
      properties: { ...feature.properties, map_color: featureColor(feature.properties ?? {}, layer) },
    })),
  };
}

export default function MapLibreSurface({
  collection, selectedLocation, layer, mapStyle, onSelectDistrict, fallback,
  showBoundaries = true, showForecastMarkers = true, forecastMarkerValue,
}: {
  collection: FeatureCollection | null;
  selectedLocation: { latitude?: number; longitude?: number } | null;
  layer: OperationalLayer;
  mapStyle: MapStyle;
  onSelectDistrict: (id: string) => void;
  fallback: ReactNode;
  showBoundaries?: boolean;
  showForecastMarkers?: boolean;
  forecastMarkerValue?: number | null;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markerRef = useRef<MapMarker | null>(null);
  const latestRef = useRef({ collection, layer, showBoundaries, onSelectDistrict });
  const [error, setError] = useState<string | null>(null);
  const selected = useMemo(() => typeof selectedLocation?.latitude === 'number' && typeof selectedLocation.longitude === 'number'
    ? { latitude: selectedLocation.latitude, longitude: selectedLocation.longitude } : null, [selectedLocation]);

  useEffect(() => { latestRef.current = { collection, layer, showBoundaries, onSelectDistrict }; }, [collection, layer, showBoundaries, onSelectDistrict]);

  useEffect(() => {
    if (!hostRef.current) return;
    let active = true;
    const map = new MapLibreMap({
      container: hostRef.current,
      style: STYLE_URL.standard,
      center: [82.5, 22.5],
      zoom: 4.2,
      minZoom: 3,
      maxZoom: 14,
      attributionControl: false,
      cooperativeGestures: true,
    });
    mapRef.current = map;
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right');
    map.addControl(new AttributionControl({ compact: true, customAttribution: '© OpenStreetMap contributors' }), 'bottom-right');
    map.getContainer().dataset.mapStyle = STYLE_URL.standard;
    let interactionBound = false;

    const updateDistrictSource = () => {
      if (!active || !map.isStyleLoaded()) return;
      const data = mapData(latestRef.current.collection, latestRef.current.layer);
      const source = map.getSource('anvaya-districts') as GeoJSONSource | undefined;
      if (!source) {
        map.addSource('anvaya-districts', { type: 'geojson', data: data as GeoJSON.FeatureCollection });
        map.addLayer({ id: 'anvaya-district-fill', type: 'fill', source: 'anvaya-districts', paint: { 'fill-color': ['get', 'map_color'], 'fill-opacity': ['case', ['==', ['get', 'forecast_status'], 'LIVE'], 0.67, 0.03] } });
        map.addLayer({ id: 'anvaya-district-line', type: 'line', source: 'anvaya-districts', paint: { 'line-color': ['case', ['==', ['get', 'selected'], true], '#12324A', '#718797'], 'line-width': ['case', ['==', ['get', 'selected'], true], 2, 0.65], 'line-opacity': latestRef.current.showBoundaries ? 0.88 : 0 } });
      } else {
        source.setData(data as GeoJSON.FeatureCollection);
        if (map.getLayer('anvaya-district-line')) map.setPaintProperty('anvaya-district-line', 'line-opacity', latestRef.current.showBoundaries ? 0.88 : 0);
        if (map.getLayer('anvaya-district-fill')) map.setPaintProperty('anvaya-district-fill', 'fill-opacity', ['case', ['==', ['get', 'forecast_status'], 'LIVE'], 0.67, latestRef.current.showBoundaries ? 0.03 : 0]);
      }
      if (map.getLayer('anvaya-district-fill')) {
        map.setPaintProperty('anvaya-district-fill', 'fill-opacity', ['case', ['==', ['get', 'forecast_status'], 'LIVE'], 0.67, latestRef.current.showBoundaries ? 0.03 : 0]);
      }
      if (!interactionBound && map.getLayer('anvaya-district-fill')) {
        interactionBound = true;
        map.on('click', 'anvaya-district-fill', (event) => {
          const id = event.features?.[0]?.properties?.anvaya_district_id;
          if (typeof id === 'string' && id) latestRef.current.onSelectDistrict(id);
        });
        map.on('mouseenter', 'anvaya-district-fill', () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mouseleave', 'anvaya-district-fill', () => { map.getCanvas().style.cursor = ''; });
      }
    };
    map.on('load', updateDistrictSource);
    map.on('style.load', updateDistrictSource);
    map.on('error', (event) => { if (active && event.error) setError('Open map tiles are currently unavailable. The sourced district map remains available below.'); });
    map.once('load', () => { if (active) setError(null); });
    return () => {
      active = false;
      markerRef.current?.remove(); markerRef.current = null;
      map.remove(); mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const styleUrl = STYLE_URL[mapStyle === 'satellite' ? 'standard' : mapStyle];
    if (map.getContainer().dataset.mapStyle !== styleUrl) {
      map.getContainer().dataset.mapStyle = styleUrl;
      map.setStyle(styleUrl);
    }
    if (map.isStyleLoaded()) {
      const source = map.getSource('anvaya-districts') as GeoJSONSource | undefined;
      if (source) source.setData(mapData(collection, layer) as GeoJSON.FeatureCollection);
      if (map.getLayer('anvaya-district-line')) map.setPaintProperty('anvaya-district-line', 'line-opacity', showBoundaries ? 0.88 : 0);
      if (map.getLayer('anvaya-district-fill')) map.setPaintProperty('anvaya-district-fill', 'fill-opacity', ['case', ['==', ['get', 'forecast_status'], 'LIVE'], 0.67, showBoundaries ? 0.03 : 0]);
    }
  }, [collection, layer, mapStyle, showBoundaries]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selected) return;
    map.flyTo({ center: [selected.longitude, selected.latitude], zoom: Math.max(map.getZoom(), 7), duration: 700 });
  }, [selected]);

  useEffect(() => {
    markerRef.current?.remove(); markerRef.current = null;
    const map = mapRef.current;
    if (!map || !selected || !showForecastMarkers || typeof forecastMarkerValue !== 'number' || !Number.isFinite(forecastMarkerValue)) return;
    const marker = new Marker({ color: '#D34F9A' })
      .setLngLat([selected.longitude, selected.latitude])
      .setPopup(new Popup({ offset: 24 }).setText(`ANVAYA returned forecast: ${forecastMarkerValue.toFixed(1)} mm / 24 h`))
      .addTo(map);
    marker.getElement().setAttribute('aria-label', `Returned ANVAYA forecast marker, ${forecastMarkerValue.toFixed(1)} millimeters`);
    markerRef.current = marker;
    return () => { marker.remove(); if (markerRef.current === marker) markerRef.current = null; };
  }, [selected, showForecastMarkers, forecastMarkerValue]);

  if (error) return <div><p role="status" className="mb-2 text-xs text-[#667b8a]">{error}</p>{fallback}</div>;
  return <div ref={hostRef} className="min-h-[360px] w-full rounded-md bg-[#edf3f6]" role="region" aria-label="OpenStreetMap-based map with ANVAYA district forecast layers" />;
}
