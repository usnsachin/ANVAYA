'use client';

import { useCallback } from 'react';
import Link from 'next/link';
import LocationSearch from '@/components/operational/LocationSearch';
import LocateMeButton from '@/components/operational/LocateMeButton';
import { asRecord, useOperationalWorkspace, type LocationSelection } from '@/components/operational/OperationalWorkspace';
import { displayNumber, displayPercent, displayText, warningCategory, warningTone, utcLabel } from '@/components/operational/operational-format';

function locationForecastValue(forecast: Record<string, unknown> | null, locationKind: 'district' | 'place' | undefined, field: string) {
  if (!forecast) return undefined;
  return locationKind === 'place' ? forecast[field] : forecast[field === 'anvaya_corrected' ? 'anvaya_rain_mm' : field];
}

function shortExplanation(forecast: Record<string, unknown> | null, regime: string) {
  if (!forecast) return 'A location-specific outlook appears here after a forecast is requested.';
  const explanation = asRecord(forecast.anvaya_explanation);
  const drivers = Array.isArray(explanation.top_feature_contributions)
    ? explanation.top_feature_contributions.filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null).slice(0, 2)
    : [];
  const names = drivers.map((item) => displayText(item.physical_driver, '')).filter(Boolean);
  if (regime && names.length) return `The model classified the outlook as ${regime.toLocaleLowerCase()}, with its correction most associated with ${names.join(' and ')}; this is attribution, not causal evidence.`;
  if (regime) return `The model classified the outlook as ${regime.toLocaleLowerCase()}; no causal explanation is implied.`;
  return 'The selected location has no returned ANVAYA forecast yet; no rainfall values are inferred.';
}

export default function HomePage() {
  const {
    boundaries, loadingBoundaries, boundaryError, selectedLocation, selectLocation,
    selectedForecast, selectedForecastStatus, selectedForecastUpdatedAt, requestLocationForecast,
    requestSelectedForecast, loading, requestError,
  } = useOperationalWorkspace();
  const onLocationSelect = useCallback((location: LocationSelection) => {
    selectLocation(location);
    void requestLocationForecast(location);
  }, [selectLocation, requestLocationForecast]);
  const locationKind = selectedLocation?.kind;
  const rawGfs = locationKind === 'place' ? asRecord(asRecord(selectedForecast?.raw_gfs).data) : {};
  const metadata = asRecord(selectedForecast?.source_metadata);
  const modelMetadata = asRecord(metadata.anvaya);
  const timeWindow = locationKind === 'place' ? asRecord(metadata.target_window) : asRecord(selectedForecast?.target_window);
  const rain = locationForecastValue(selectedForecast, locationKind, 'anvaya_corrected');
  const rawRain = locationKind === 'place' ? rawGfs.gfs_daily_rain_mm : selectedForecast?.raw_gfs_rain_mm;
  const correction = locationKind === 'place' ? selectedForecast?.anvaya_minus_raw_mm : selectedForecast?.anvaya_minus_global_mm;
  const warning = locationKind === 'place' ? asRecord(selectedForecast?.district_risk).warning_level ?? selectedForecast?.warning_level : selectedForecast?.warning_level;
  const regime = displayText(selectedForecast?.regime, 'Not returned');
  const heavyProbability = selectedForecast?.heavy_probability;
  const sourceStatuses = (['gfs', 'gefs', 'ecmwf', 'imerg'] as const).map((source) => ({
    source,
    status: displayText(asRecord(metadata[source]).status, 'UNAVAILABLE'),
  }));
  const isLive = selectedForecastStatus === 'LIVE';
  const hasForecastResponse = selectedForecast !== null;
  const updatedText = selectedForecastUpdatedAt
    ? new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Kolkata' }).format(new Date(selectedForecastUpdatedAt))
    : null;

  return <div className="space-y-5 pb-5">
    <header className="grid gap-5 border-b border-[#d9e2e8] pb-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(360px,1.1fr)] lg:items-end">
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#667b8a]">HOME · LOCAL OUTLOOK</p>
        <h1 className="mt-1.5 text-[25px] font-semibold leading-tight tracking-tight text-[#12324A] sm:text-[30px]">Rainfall where you are</h1>
        <p className="mt-1.5 max-w-lg text-sm leading-relaxed text-[#536a7a]">Start with one place. Forecast values are shown only when returned by the operational service.</p>
      </div>
      <div className="rounded-md border border-[#d9e2e8] bg-white p-4 shadow-[0_2px_10px_rgba(18,50,74,0.035)]">
        {loadingBoundaries && !boundaries ? <div className="h-11 animate-pulse rounded bg-[#eef3f6]" aria-label="Loading district directory" /> : <><LocationSearch districts={boundaries?.features ?? []} selectedLocation={selectedLocation} onSelect={onLocationSelect} /><div className="mt-2"><LocateMeButton /></div></>}
        {boundaryError && <p role="status" className="mt-2 text-xs text-[#8a5a1f]">{boundaryError}</p>}
      </div>
    </header>

    <section aria-label="Selected location rainfall summary" className="overflow-hidden rounded-md border border-[#d6e0e6] bg-white shadow-[0_4px_18px_rgba(18,50,74,0.045)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#e4ebef] px-4 py-3 sm:px-6">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#667b8a]">LOCATION</p>
          <h2 className="mt-0.5 truncate text-base font-semibold text-[#12324A]">{selectedLocation ? `${selectedLocation.name}${selectedLocation.state ? `, ${selectedLocation.state}` : ''}` : loadingBoundaries ? 'Loading location directory…' : 'Choose a location'}</h2>
          {selectedLocation?.kind === 'place' && selectedLocation.address && <p className="mt-0.5 truncate text-xs text-[#667b8a]">{selectedLocation.address}</p>}
        </div>
        <StatusPill status={hasForecastResponse ? selectedForecastStatus : 'REFERENCE'} label={hasForecastResponse ? selectedForecastStatus : 'FORECAST NOT REQUESTED'} />
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1.2fr)_minmax(250px,0.8fr)]">
        <div className="min-w-0 px-4 py-5 sm:px-6 sm:py-6">
          <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#536a7a]">ANVAYA · 24-HOUR RAINFALL</p>
          <div className="mt-2 flex flex-wrap items-end gap-x-3 gap-y-1">
            <p className="text-[52px] font-semibold leading-none tracking-[-0.045em] tabular-nums text-[#12324A] sm:text-[64px]">{isLive ? displayNumber(rain, '', 1) : '—'}<span className="ml-2 text-[24px] font-medium tracking-normal text-[#536a7a]">mm</span></p>
            <span className="pb-1 text-xs text-[#667b8a]">24-hour accumulation</span>
          </div>
          <p className="mt-2 text-xs text-[#667b8a]">{timeWindow.start || timeWindow.end ? `${utcLabel(timeWindow.start)} → ${utcLabel(timeWindow.end)}` : 'Forecast window is not available until a response is returned.'}</p>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[#536a7a]">{shortExplanation(selectedForecast, regime === 'Not returned' ? '' : regime)}</p>
          {requestError && <p role="alert" className="mt-3 rounded-md border border-[#f0caca] bg-[#fff5f5] px-3 py-2 text-xs text-[#963232]">{requestError}</p>}
          {!isLive && <p className="mt-3 rounded-md border border-[#e2e9ed] bg-[#f6f9fa] px-3 py-2 text-xs leading-relaxed text-[#536a7a]">{hasForecastResponse ? 'Location lookup available. Forecast requires spatially resolved ANVAYA data.' : 'No forecast has been requested for this location. Nothing is estimated or filled from a preset.'}</p>}
          <div className="mt-5 flex flex-wrap items-center gap-3">
            {!isLive && selectedLocation && <button type="button" disabled={loading} onClick={() => void requestSelectedForecast()} className="min-h-10 rounded-md bg-[#1769AA] px-4 text-xs font-semibold text-white hover:bg-[#12598f] disabled:cursor-wait disabled:opacity-60">{loading ? 'Requesting outlook…' : hasForecastResponse ? 'Retry forecast' : 'Get location outlook'}</button>}
            {selectedLocation && <><Link href="/map" className="inline-flex min-h-10 items-center rounded-md border border-[#b7c9d5] px-4 text-xs font-semibold text-[#1769AA] hover:bg-[#f1f7fb]">VIEW MAP <span aria-hidden="true" className="ml-2">→</span></Link><Link href="/alerts" className="inline-flex min-h-10 items-center rounded-md border border-[#b7c9d5] px-4 text-xs font-semibold text-[#1769AA] hover:bg-[#f1f7fb]">VIEW ALERTS <span aria-hidden="true" className="ml-2">→</span></Link></>}
          </div>
        </div>

        <div className="border-t border-[#e4ebef] bg-[#f8fafb] p-4 sm:p-6 lg:border-l lg:border-t-0">
          <div className="flex items-center justify-between gap-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#667b8a]">ANVAYA DECISION RULE</p>
            <span className={`rounded-sm border px-2 py-1 text-[9px] font-bold tracking-wide ${warningTone(warning)}`}>{isLive ? warningCategory(warning) : '—'}</span>
          </div>
          <p className={`mt-2 text-lg font-semibold ${isLive ? 'text-[#12324A]' : 'text-[#667b8a]'}`}>{isLive ? displayText(warningCategory(warning), 'UNAVAILABLE') : 'Risk not assessed'}</p>
          <p className="mt-1 text-xs text-[#667b8a]">A system decision category, not an official IMD warning.</p>
          <dl className="mt-5 divide-y divide-[#dfe7ec] border-y border-[#dfe7ec]">
            <SummaryRow label="Heavy-rain probability" value={isLive ? displayPercent(heavyProbability) : '—'} />
            <SummaryRow label="Weather regime" value={isLive ? displayText(selectedForecast?.regime) : '—'} />
            <SummaryRow label="Correction" value={isLive ? displayNumber(correction, ' mm', 1) : '—'} />
            <SummaryRow label="Source agreement" value={isLive ? `${displayPercent(asRecord(selectedForecast?.forecast_confidence).score)} · uncalibrated` : '—'} />
          </dl>
          <p className="mt-3 text-[10px] leading-relaxed text-[#667b8a]">Correction is ANVAYA minus {locationKind === 'place' ? 'raw GFS' : 'Global ML'}, as returned by this forecast. It is not an independently measured rainfall difference.</p>
          <details className="mt-4 rounded-sm border border-[#d9e2e8] bg-white px-3 py-2.5">
            <summary className="cursor-pointer text-xs font-semibold text-[#1769AA]">Forecast details & sources</summary>
            <div className="mt-3 space-y-3 border-t border-[#e7edf1] pt-3">
              <SummaryRow label="Raw GFS · 24 h" value={isLive ? displayNumber(rawRain, ' mm', 1) : '—'} />
              <SummaryRow label="Updated" value={updatedText || 'Not yet returned'} />
              <div className="grid grid-cols-2 gap-2">{sourceStatuses.map(({ source, status }) => <div key={source} className="flex items-center justify-between gap-2 rounded-sm bg-[#f5f8fa] px-2.5 py-2 text-[10px]"><span className="uppercase text-[#536a7a]">{source}</span><span className={`font-semibold ${status === 'LIVE' ? 'text-[#2E8B57]' : 'text-[#667b8a]'}`}>{status}</span></div>)}</div>
              {locationKind === 'place' && <p className="text-[10px] leading-relaxed text-[#667b8a]">This is a coordinate-resolved point forecast. It is not represented as a village-wide or locality-wide area average.</p>}
              {typeof asRecord(modelMetadata).error === 'string' && <p className="text-xs text-[#963232]">{String(modelMetadata.error)}</p>}
            </div>
          </details>
        </div>
      </div>
    </section>

    <section aria-label="Forecast time checkpoints" className="overflow-hidden rounded-md border border-[#d9e2e8] bg-white">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#e4ebef] px-4 py-3 sm:px-5"><h2 className="text-xs font-semibold text-[#12324A]">Forecast checkpoints</h2><p className="text-[10px] text-[#667b8a]">The current API returns a 24-hour accumulation, not an hourly curve.</p></div>
      <div className="grid grid-cols-3 divide-x divide-[#e4ebef]">
        <Checkpoint label="NOW" value="Not returned" />
        <Checkpoint label="12 HOURS" value="Not returned" />
        <Checkpoint label="24 HOURS" value={isLive ? displayNumber(rain, ' mm', 1) : '—'} detail={isLive ? 'Total for the stated 24-hour window' : 'Awaiting a forecast'} />
      </div>
    </section>

    <p className="text-[10px] leading-relaxed text-[#667b8a]">Operational forecasts use live external NWP sources and are not part of the frozen Phase 3F scientific evaluation. Location resolution does not guarantee that ANVAYA has a valid forecast for that point.</p>
  </div>;
}

function StatusPill({ status, label }: { status: string; label: string }) {
  const tone = status === 'LIVE' ? 'border-[#b9ddc8] bg-[#EAF5EE] text-[#246b43]' : status === 'REFERENCE' ? 'border-[#c9dbe8] bg-[#eff6fb] text-[#1769AA]' : 'border-[#d9e2e8] bg-[#f1f4f6] text-[#667b8a]';
  return <span className={`rounded-sm border px-2.5 py-1.5 text-[9px] font-bold tracking-[0.08em] ${tone}`}>{label}</span>;
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return <div className="flex items-start justify-between gap-3 py-2.5 text-xs"><dt className="text-[#667b8a]">{label}</dt><dd className="max-w-[62%] text-right font-semibold tabular-nums text-[#12324A]">{value}</dd></div>;
}

function Checkpoint({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return <div className="min-w-0 px-3 py-3.5 sm:px-5"><p className="text-[9px] font-semibold tracking-[0.12em] text-[#667b8a]">{label}</p><p className="mt-1 truncate text-sm font-semibold tabular-nums text-[#12324A]">{value}</p>{detail && <p className="mt-1 text-[10px] leading-relaxed text-[#667b8a]">{detail}</p>}</div>;
}
