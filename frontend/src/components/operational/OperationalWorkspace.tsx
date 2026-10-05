'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { fetchOperationalForecast, fetchOperationalMap } from '@/lib/api';
import type { DistrictFeature, FeatureCollection } from '@/components/district/DistrictBoundaryMap';

export type DataRecord = Record<string, unknown>;
export type DistrictForecast = DataRecord & { district?: DataRecord };
export type OperationalLayer = 'rain' | 'risk' | 'regime' | 'correction' | 'raw_nwp' | 'global_ml';
export type MapStyle = 'standard' | 'light' | 'dark' | 'satellite';
export type LocationSelection = {
  key: string;
  kind: 'district' | 'place';
  name: string;
  state?: string;
  country?: string;
  admin1?: string;
  admin2?: string;
  address?: string;
  districtId?: string;
  latitude?: number;
  longitude?: number;
};

type WorkspaceValue = {
  boundaries: FeatureCollection | null;
  visibleCollection: FeatureCollection | null;
  loadingBoundaries: boolean;
  boundaryError: string | null;
  computedDistricts: DistrictForecast[];
  selectedDistrictId: string;
  selectedLocation: LocationSelection | null;
  selectLocation: (location: LocationSelection) => void;
  selectDistrict: (id: string) => void;
  selectedForecast: DataRecord | null;
  selectedForecastStatus: 'LIVE' | 'UNAVAILABLE' | 'REFERENCE';
  selectedForecastUpdatedAt: string | null;
  layer: OperationalLayer;
  setLayer: (layer: OperationalLayer) => void;
  mapStyle: MapStyle;
  setMapStyle: (style: MapStyle) => void;
  loading: boolean;
  loadingDistrictId: string | null;
  requestError: string | null;
  findFeature: (id: string) => DistrictFeature | undefined;
  getForecast: (id?: string) => DistrictForecast | undefined;
  requestDistrictForecast: (id?: string) => Promise<DistrictForecast | null>;
  requestLocationForecast: (location: LocationSelection) => Promise<DataRecord | null>;
  requestSelectedForecast: () => Promise<DataRecord | null>;
};

const WorkspaceContext = createContext<WorkspaceValue | null>(null);

export function asRecord(value: unknown): DataRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as DataRecord : {};
}

function districtId(row: DistrictForecast) {
  return String(asRecord(row.district).district_id ?? '');
}

function numericValue(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function mapValue(row: DistrictForecast, layer: OperationalLayer) {
  switch (layer) {
    case 'rain': return numericValue(row.anvaya_rain_mm);
    case 'raw_nwp': return numericValue(row.raw_gfs_rain_mm);
    case 'global_ml': return numericValue(row.global_ml_rain_mm);
    case 'correction': return numericValue(row.anvaya_minus_global_mm);
    case 'regime': return typeof row.regime === 'string' ? row.regime : null;
    case 'risk': return typeof row.warning_level === 'string' ? row.warning_level : null;
  }
}

function featureLocation(feature: DistrictFeature, id: string): LocationSelection {
  const properties = feature.properties || {};
  return {
    key: `district:${id}`, kind: 'district', districtId: id,
    name: String(properties.district ?? 'District'),
    state: String(properties.state_name ?? '') || undefined,
  };
}

export function OperationalWorkspaceProvider({ children }: { children: ReactNode }) {
  const [boundaries, setBoundaries] = useState<FeatureCollection | null>(null);
  const [loadingBoundaries, setLoadingBoundaries] = useState(true);
  const [boundaryError, setBoundaryError] = useState<string | null>(null);
  const [computedDistricts, setComputedDistricts] = useState<DistrictForecast[]>([]);
  const [pointForecasts, setPointForecasts] = useState<Record<string, DataRecord>>({});
  const [selectedDistrictId, setSelectedDistrictId] = useState('');
  const [selectedLocation, setSelectedLocation] = useState<LocationSelection | null>(null);
  const [layer, setLayer] = useState<OperationalLayer>('risk');
  const [mapStyle, setMapStyle] = useState<MapStyle>('standard');
  const [loadingLocationKey, setLoadingLocationKey] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchOperationalMap(undefined, 'district_warning')
      .then((result: unknown) => {
        if (!active) return;
        const payload = asRecord(result);
        const features = Array.isArray(payload.features) ? payload.features as DistrictFeature[] : [];
        const collection: FeatureCollection = { type: 'FeatureCollection', features };
        setBoundaries(collection);
        const wayanad = features.find((feature) => String(feature.properties?.district ?? '').toLowerCase() === 'wayanad');
        const defaultId = String(wayanad?.properties?.anvaya_district_id ?? '');
        if (wayanad && defaultId) {
          setSelectedDistrictId(defaultId);
          setSelectedLocation(featureLocation(wayanad, defaultId));
        }
        if (!features.length) setBoundaryError('The district reference catalog returned no boundaries.');
      })
      .catch((error: unknown) => {
        if (active) setBoundaryError(error instanceof Error ? error.message : 'District reference catalog is unavailable.');
      })
      .finally(() => { if (active) setLoadingBoundaries(false); });
    return () => { active = false; };
  }, []);

  const findFeature = useCallback((id: string) => boundaries?.features.find((feature) => String(feature.properties?.anvaya_district_id ?? '') === id), [boundaries]);
  const getForecast = useCallback((id = selectedDistrictId) => computedDistricts.find((row) => districtId(row) === id), [computedDistricts, selectedDistrictId]);

  const selectLocation = useCallback((location: LocationSelection) => {
    setSelectedLocation(location);
    setSelectedDistrictId(location.kind === 'district' ? location.districtId || '' : '');
    setRequestError(null);
  }, []);

  const selectDistrict = useCallback((id: string) => {
    const feature = findFeature(id);
    if (feature) selectLocation(featureLocation(feature, id));
  }, [findFeature, selectLocation]);

  const requestDistrictForecast = useCallback(async (id = selectedDistrictId) => {
    if (!id || loadingLocationKey) return null;
    selectDistrict(id);
    const locationKey = `district:${id}`;
    setLoadingLocationKey(locationKey);
    setRequestError(null);
    try {
      const result = asRecord(await fetchOperationalMap(id, 'district_warning'));
      const summary = asRecord(result.summary) as DistrictForecast;
      if (!Object.keys(asRecord(summary.district)).length) {
        throw new Error('The district service returned no forecast summary. No fallback values were substituted.');
      }
      setComputedDistricts((current) => [summary, ...current.filter((row) => districtId(row) !== id)]);
      return summary;
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Selected district forecast is unavailable.');
      return null;
    } finally {
      setLoadingLocationKey(null);
    }
  }, [loadingLocationKey, selectedDistrictId, selectDistrict]);

  const requestPlaceForecast = useCallback(async (location: LocationSelection) => {
    if (loadingLocationKey) return null;
    const latitude = location.latitude;
    const longitude = location.longitude;
    if (typeof latitude !== 'number' || typeof longitude !== 'number' || latitude < 8 || latitude > 37 || longitude < 68 || longitude > 97) {
      setRequestError('Location lookup available. Forecast requires spatially resolved ANVAYA data within the supported India domain.');
      return null;
    }
    setLoadingLocationKey(location.key);
    setRequestError(null);
    try {
      const response = asRecord(await fetchOperationalForecast(latitude, longitude));
      if (!Object.keys(response).length) throw new Error('The forecast service returned an empty response.');
      setPointForecasts((current) => ({ ...current, [location.key]: response }));
      if (response.anvaya_corrected == null && asRecord(response.source_metadata).anvaya) {
        const error = asRecord(asRecord(response.source_metadata).anvaya).error;
        if (typeof error === 'string' && error) setRequestError(error);
      }
      return response;
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Resolved location forecast is unavailable.');
      return null;
    } finally {
      setLoadingLocationKey(null);
    }
  }, [loadingLocationKey]);

  const selectedForecast = useMemo(() => {
    if (!selectedLocation) return null;
    return selectedLocation.kind === 'district'
      ? getForecast(selectedLocation.districtId) ?? null
      : pointForecasts[selectedLocation.key] || null;
  }, [selectedLocation, getForecast, pointForecasts]);

  const selectedForecastStatus = useMemo<'LIVE' | 'UNAVAILABLE' | 'REFERENCE'>(() => {
    if (!selectedForecast) return 'REFERENCE';
    if (selectedLocation?.kind === 'district') return selectedForecast.status === 'LIVE' ? 'LIVE' : 'UNAVAILABLE';
    const modelStatus = asRecord(asRecord(selectedForecast.source_metadata).anvaya).status;
    return modelStatus === 'LIVE' && selectedForecast.anvaya_corrected != null ? 'LIVE' : 'UNAVAILABLE';
  }, [selectedForecast, selectedLocation]);

  const selectedForecastUpdatedAt = typeof selectedForecast?.timestamp === 'string' ? selectedForecast.timestamp : null;

  const requestLocationForecast = useCallback((location: LocationSelection) => location.kind === 'district'
    ? requestDistrictForecast(location.districtId)
    : requestPlaceForecast(location), [requestDistrictForecast, requestPlaceForecast]);

  const requestSelectedForecast = useCallback(() => selectedLocation
    ? requestLocationForecast(selectedLocation)
    : Promise.resolve(null), [selectedLocation, requestLocationForecast]);

  const visibleCollection = useMemo<FeatureCollection | null>(() => {
    if (!boundaries) return null;
    const rowsById = new Map(computedDistricts.map((row) => [districtId(row), row]));
    return {
      ...boundaries,
      features: boundaries.features.map((feature) => {
        const properties = feature.properties || {};
        const id = String(properties.anvaya_district_id ?? '');
        const row = rowsById.get(id);
        if (!row) return feature;
        return {
          ...feature,
          properties: {
            ...properties,
            forecast_value: mapValue(row, layer),
            forecast_layer: layer,
            forecast_status: row.status ?? 'UNAVAILABLE',
            warning_level: row.warning_level ?? 'UNAVAILABLE',
            regime: row.regime ?? null,
          },
        };
      }),
    };
  }, [boundaries, computedDistricts, layer]);

  const value = useMemo<WorkspaceValue>(() => ({
    boundaries, visibleCollection, loadingBoundaries, boundaryError, computedDistricts,
    selectedDistrictId, selectedLocation, selectLocation, selectDistrict, selectedForecast,
    selectedForecastStatus, selectedForecastUpdatedAt, layer, setLayer, mapStyle, setMapStyle,
    loading: loadingLocationKey !== null,
    loadingDistrictId: loadingLocationKey?.startsWith('district:') ? loadingLocationKey.slice('district:'.length) : null,
    requestError, findFeature, getForecast, requestDistrictForecast, requestLocationForecast, requestSelectedForecast,
  }), [
    boundaries, visibleCollection, loadingBoundaries, boundaryError, computedDistricts,
    selectedDistrictId, selectedLocation, selectLocation, selectDistrict, selectedForecast,
    selectedForecastStatus, selectedForecastUpdatedAt, layer, mapStyle, loadingLocationKey,
    requestError, findFeature, getForecast, requestDistrictForecast, requestLocationForecast, requestSelectedForecast,
  ]);

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useOperationalWorkspace() {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error('useOperationalWorkspace must be used within OperationalWorkspaceProvider');
  return value;
}
