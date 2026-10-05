'use client';

import dynamic from 'next/dynamic';
import { useCallback, useState } from 'react';
import DistrictBoundaryMap, { regimeColor } from '@/components/district/DistrictBoundaryMap';
import LocationSearch from './LocationSearch';
import LocateMeButton from './LocateMeButton';
import { asRecord, useOperationalWorkspace, type LocationSelection, type OperationalLayer } from './OperationalWorkspace';
import { displayNumber, displayPercent, displayText, warningCategory, warningTone, utcLabel } from './operational-format';

const LAYERS: { id: OperationalLayer; label: string }[] = [
  { id: 'rain', label: 'ANVAYA' },
  { id: 'global_ml', label: 'GLOBAL ML' },
  { id: 'risk', label: 'RISK' },
  { id: 'regime', label: 'REGIME' },
  { id: 'correction', label: 'ANVAYA CORRECTION' },
  { id: 'raw_nwp', label: 'RAW NWP' },
];
const MAP_STYLES = [
  { id: 'standard', label: 'STANDARD', available: true },
  { id: 'light', label: 'LIGHT', available: true },
  { id: 'dark', label: 'DARK', available: true },
  { id: 'satellite', label: 'SATELLITE', available: false },
] as const;

const MapLibreSurface = dynamic(() => import('./MapLibreSurface'), { ssr: false });

function legend(layer: OperationalLayer, regimes: string[]) {
  if (layer === 'risk') return [
    ['#2E8B57', 'LOW'], ['#E9B949', 'MODERATE'], ['#D64545', 'HEAVY / SEVERE'], ['#D34F9A', 'EXCEPTIONAL / EXTREME'],
  ] as const;
  if (layer === 'regime') return regimes.length
    ? regimes.map((regime) => [regimeColor(regime), regime] as const)
    : [['#68859A', 'No returned regime values']] as const;
  if (layer === 'correction') return [['#1769AA', 'ANVAYA below Global ML'], ['#79909F', 'No difference'], ['#D64545', 'ANVAYA above Global ML']] as const;
  return [['#2E8B57', '<64.5 mm'], ['#E9B949', '64.5–115.5 mm'], ['#D64545', '115.6–204.4 mm'], ['#D34F9A', '≥204.5 mm']] as const;
}

export default function OperationalMapSurface() {
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [showForecastMarkers, setShowForecastMarkers] = useState(true);
  const {
    boundaries, visibleCollection, loadingBoundaries, boundaryError, layer, setLayer, mapStyle, setMapStyle,
    selectedDistrictId, selectedLocation, selectLocation, selectedForecast, selectedForecastStatus,
    computedDistricts, findFeature, requestDistrictForecast, requestLocationForecast,
    loading, requestError,
  } = useOperationalWorkspace();
  const selectAndRequest = useCallback((location: LocationSelection) => {
    selectLocation(location);
    void requestLocationForecast(location);
  }, [selectLocation, requestLocationForecast]);
  const selectDistrictAndRequest = useCallback((id: string) => {
    const feature = findFeature(id);
    if (!feature) return;
    void requestDistrictForecast(id);
  }, [findFeature, requestDistrictForecast]);
  const regimes = [...new Set(computedDistricts.map((row) => displayText(row.regime, '')).filter(Boolean))];
  const pointCoordinate = selectedLocation?.kind === 'place' && typeof selectedLocation.latitude === 'number' && typeof selectedLocation.longitude === 'number'
    ? { latitude: selectedLocation.latitude, longitude: selectedLocation.longitude }
    : null;
  const selectedFeature = selectedDistrictId ? findFeature(selectedDistrictId) : undefined;
  const forecast = selectedForecast || undefined;
  const markerForecast = typeof selectedForecast?.anvaya_corrected === 'number' ? selectedForecast.anvaya_corrected : null;

  return <section aria-label="India rainfall intelligence map" className="overflow-hidden rounded-md border border-[#d5e0e6] bg-white shadow-[0_4px_18px_rgba(18,50,74,0.045)]">
    <div className="grid gap-3 border-b border-[#e4ebef] p-3 sm:p-4 lg:grid-cols-[minmax(260px,0.72fr)_minmax(0,1.28fr)] lg:items-end">
      <div><LocationSearch districts={boundaries?.features ?? []} selectedLocation={selectedLocation} onSelect={selectAndRequest} compact label="Find a place, district, village or locality" /><div className="mt-2"><LocateMeButton /></div></div>
      <div className="min-w-0">
        <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-[#667b8a]">MAP LAYER</p>
        <div className="flex gap-1 overflow-x-auto pb-1" role="group" aria-label="Rainfall map layer">
          {LAYERS.map((item) => <button key={item.id} type="button" aria-pressed={layer === item.id} onClick={() => setLayer(item.id)} className={`min-h-9 shrink-0 rounded-sm px-2.5 text-[9px] font-semibold tracking-[0.04em] transition-colors ${layer === item.id ? 'bg-[#1769AA] text-white' : 'border border-[#d9e2e8] bg-white text-[#536a7a] hover:bg-[#f1f7fb]'}`}>{item.label}</button>)}
        </div>
      </div>
    </div>

    <div className="grid xl:grid-cols-[minmax(0,1.55fr)_minmax(300px,0.72fr)]">
      <div className="min-w-0 p-3 sm:p-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1.5" role="group" aria-label="Map base style">
            {MAP_STYLES.map((style) => <button key={style.id} type="button" aria-pressed={mapStyle === style.id} disabled={!style.available} title={!style.available ? 'Satellite imagery is unavailable from the configured keyless map source' : undefined}
              onClick={() => setMapStyle(style.id)} className={`min-h-8 rounded-sm px-2 text-[8px] font-semibold tracking-wide disabled:cursor-not-allowed disabled:opacity-35 ${mapStyle === style.id ? 'bg-[#12324A] text-white' : 'border border-[#d9e2e8] bg-white text-[#536a7a] hover:bg-[#f1f7fb]'}`}>{style.label}</button>)}
          </div>
          <div className="flex items-center gap-3 text-[9px] text-[#667b8a]">
            <label className="inline-flex items-center gap-1.5"><input type="checkbox" checked={showBoundaries} onChange={(event) => setShowBoundaries(event.target.checked)} />DISTRICT BOUNDARIES</label>
            <label className="inline-flex items-center gap-1.5"><input type="checkbox" checked={showForecastMarkers} onChange={(event) => setShowForecastMarkers(event.target.checked)} />FORECAST MARKERS</label>
            <span>733 boundaries · REFERENCE</span>
          </div>
        </div>

        {loadingBoundaries ? <div className="flex min-h-[360px] items-center justify-center rounded-md border border-dashed border-[#c8d5de] bg-[#f6f9fa] text-sm text-[#667b8a]">Loading India district boundaries…</div>
          : boundaryError && !boundaries ? <div role="alert" className="flex min-h-[360px] items-center justify-center rounded-md border border-dashed border-[#e7b7b7] bg-[#fff5f5] px-5 text-center text-sm text-[#963232]">{boundaryError}</div>
            : <MapLibreSurface collection={visibleCollection} selectedLocation={pointCoordinate} layer={layer} mapStyle={mapStyle} onSelectDistrict={selectDistrictAndRequest} showBoundaries={showBoundaries} showForecastMarkers={showForecastMarkers} forecastMarkerValue={markerForecast} fallback={<DistrictBoundaryMap collection={visibleCollection} selectedDistrictId={selectedDistrictId} onSelect={selectDistrictAndRequest} selectedCoordinate={pointCoordinate} mapStyle={mapStyle === 'dark' ? 'dark' : 'standard'} layer={layer} showBoundaries={showBoundaries} />} />}

        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-2 border-t border-[#e4ebef] pt-3">
          <div className="flex flex-wrap gap-x-4 gap-y-2" aria-label={`${layer} map legend`}>
            {legend(layer, regimes).map(([color, label]) => <span key={label} className="inline-flex items-center gap-1.5 text-[9px] text-[#536a7a]"><i className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />{label}</span>)}
          </div>
          <span className="text-[9px] text-[#667b8a]">{computedDistricts.length} districts queried this session</span>
        </div>
        {(layer === 'risk' || layer === 'rain' || layer === 'raw_nwp') && <p className="mt-2 text-[9px] leading-relaxed text-[#667b8a]">Threshold bands follow the configured rainfall/warning policy. Neutral districts are reference geometry, not a low-risk assessment.</p>}
        <p className="mt-2 text-[9px] leading-relaxed text-[#667b8a]">Map colors are ANVAYA visualization categories, not official IMD warning classifications. Satellite imagery is unavailable from the selected keyless source and is not fabricated. © OpenStreetMap contributors.</p>
      </div>

      <aside className="border-t border-[#e4ebef] bg-[#f8fafb] p-4 sm:p-5 xl:border-l xl:border-t-0">
        <MapLocationPanel
          location={selectedLocation}
          districtName={typeof selectedFeature?.properties?.district === 'string' ? selectedFeature.properties.district : undefined}
          districtState={typeof selectedFeature?.properties?.state_name === 'string' ? selectedFeature.properties.state_name : undefined}
          forecast={forecast}
          status={selectedForecastStatus}
          loading={loading}
          error={requestError}
          onRequest={() => selectedLocation && void requestLocationForecast(selectedLocation)}
        />
      </aside>
    </div>
  </section>;
}

function MapLocationPanel({
  location, districtName, districtState, forecast, status, loading, error, onRequest,
}: {
  location: LocationSelection | null;
  districtName?: string;
  districtState?: string;
  forecast?: Record<string, unknown>;
  status: string;
  loading: boolean;
  error: string | null;
  onRequest: () => void;
}) {
  const isPlace = location?.kind === 'place';
  const sourceMetadata = asRecord(forecast?.source_metadata);
  const rawGfs = isPlace ? asRecord(asRecord(forecast?.raw_gfs).data) : {};
  const warning = isPlace ? asRecord(forecast?.district_risk).warning_level ?? forecast?.warning_level : forecast?.warning_level;
  const rain = isPlace ? forecast?.anvaya_corrected : forecast?.anvaya_rain_mm;
  const correction = isPlace ? forecast?.anvaya_minus_global_mm ?? forecast?.anvaya_minus_raw_mm : forecast?.anvaya_minus_global_mm;
  const rawRain = isPlace ? rawGfs.gfs_daily_rain_mm : forecast?.raw_gfs_rain_mm;
  const window = isPlace ? asRecord(sourceMetadata.target_window) : asRecord(forecast?.target_window);
  const updatedAt = typeof forecast?.timestamp === 'string' ? utcLabel(forecast.timestamp) : 'Not returned';
  const live = status === 'LIVE';
  const title = location?.name || districtName || 'Select a location';
  const subtitle = location?.state || districtState || location?.address || (location ? 'India' : 'Choose a boundary or search for a place');

  return <div className="space-y-4">
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0"><p className="text-[9px] font-semibold uppercase tracking-[0.13em] text-[#667b8a]">SELECTED PLACE · {isPlace ? 'POINT' : location ? 'DISTRICT' : 'REFERENCE'}</p><h2 className="mt-1 truncate text-lg font-semibold text-[#12324A]">{title}</h2><p className="mt-0.5 text-xs text-[#667b8a]">{subtitle}</p></div>
      <span className={`shrink-0 rounded-sm border px-2 py-1 text-[8px] font-bold tracking-wide ${live ? 'border-[#b9ddc8] bg-[#EAF5EE] text-[#246b43]' : 'border-[#d9e2e8] bg-[#f1f4f6] text-[#667b8a]'}`}>{forecast ? status : 'REFERENCE'}</span>
    </div>

    {!forecast ? <div className="rounded-sm border border-dashed border-[#c8d5de] bg-white p-4">
      <p className="text-xs leading-relaxed text-[#536a7a]">{location ? 'Location lookup is available. Request its forecast to display returned values.' : 'Choose an administrative boundary or resolve a place. Unqueried areas remain neutral.'}</p>
      {location && <button type="button" onClick={onRequest} disabled={loading} className="mt-4 min-h-10 w-full rounded-sm bg-[#1769AA] px-3 text-xs font-semibold text-white hover:bg-[#12598f] disabled:opacity-50">{loading ? 'Requesting forecast…' : 'Load location forecast'}</button>}
      {error && <p role="alert" className="mt-3 break-words text-xs text-[#963232]">{error}</p>}
    </div> : <>
      <div className={`rounded-sm border p-3.5 ${warningTone(warning)}`}>
        <p className="text-[9px] font-bold uppercase tracking-[0.1em] opacity-75">ANVAYA DECISION SUPPORT · NOT AN OFFICIAL IMD WARNING</p>
        <p className="mt-1 text-sm font-semibold">{live ? warningCategory(warning) : 'Forecast unavailable'}</p>
      </div>
      <dl className="divide-y divide-[#dfe7ec] border-y border-[#dfe7ec]">
        <MapMetric label="Rainfall · 24 h" value={live ? displayNumber(rain, ' mm', 1) : '—'} />
        <MapMetric label="Heavy probability" value={live ? displayPercent(forecast.heavy_probability) : '—'} />
        <MapMetric label="Weather regime" value={live ? displayText(forecast.regime) : '—'} />
        <MapMetric label="ANVAYA correction" value={live ? displayNumber(correction, ' mm', 1) : '—'} />
        <MapMetric label="Raw NWP · GFS" value={live ? displayNumber(rawRain, ' mm', 1) : '—'} />
        <MapMetric label="Global ML" value={live ? displayNumber(forecast.global_ml_rain_mm, ' mm', 1) : '—'} />
      </dl>
      <p className="text-[10px] text-[#667b8a]">Target: {window.start || window.end ? `${utcLabel(window.start)} → ${utcLabel(window.end)}` : 'Not returned'}<br />Updated: {updatedAt}</p>
      {isPlace && <p className="rounded-sm border border-[#d9e2e8] bg-white px-3 py-2.5 text-[10px] leading-relaxed text-[#536a7a]">This is a point forecast at resolved coordinates—not a village, locality or district area average.</p>}
      <details className="rounded-sm border border-[#d9e2e8] bg-white px-3 py-2.5">
        <summary className="cursor-pointer text-xs font-semibold text-[#1769AA]">Sources & method</summary>
        <div className="mt-3 space-y-2 border-t border-[#e7edf1] pt-3 text-[10px] text-[#536a7a]">
          <p>Forecast confidence: {displayPercent(asRecord(forecast.forecast_confidence).score)} · uncalibrated source-stability diagnostic.</p>
          {(['gfs', 'gefs', 'ecmwf', 'imerg'] as const).map((source) => <div key={source} className="flex justify-between gap-2"><span className="uppercase">{source}</span><span className="font-semibold">{displayText(asRecord(sourceMetadata[source]).status, 'UNAVAILABLE')}</span></div>)}
          <p>Model attribution describes predictive associations; it is not causal evidence.</p>
        </div>
      </details>
      <button type="button" onClick={onRequest} disabled={loading} className="min-h-9 w-full rounded-sm border border-[#b7c9d5] px-3 text-xs font-semibold text-[#1769AA] hover:bg-white disabled:opacity-50">{loading ? 'Refreshing…' : 'Refresh this location'}</button>
      {error && <p role="alert" className="break-words text-xs text-[#963232]">{error}</p>}
    </>}
  </div>;
}

function MapMetric({ label, value }: { label: string; value: string }) {
  return <div className="flex items-start justify-between gap-3 py-2.5 text-xs"><dt className="text-[#667b8a]">{label}</dt><dd className="text-right font-semibold tabular-nums text-[#12324A]">{value}</dd></div>;
}
