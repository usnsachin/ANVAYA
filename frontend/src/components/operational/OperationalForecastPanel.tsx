'use client';

import { useState } from 'react';
import type { FormEvent } from 'react';
import { fetchOperationalForecast } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

type ApiRecord = Record<string, unknown>;
type ForecastPayload = ApiRecord;

function asRecord(value: unknown): ApiRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as ApiRecord
    : {};
}

function asRecords(value: unknown): ApiRecord[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function asString(value: unknown, fallback = '—') {
  return typeof value === 'string' ? value : fallback;
}

function numberText(value: unknown, digits = 1, suffix = '') {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : '—';
}

function probabilityText(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

function sourceStatus(value: unknown) {
  return asString(asRecord(value).status, 'UNAVAILABLE');
}

export default function OperationalForecastPanel({ title = 'Live point forecast' }: { title?: string }) {
  const [latitude, setLatitude] = useState('20.0');
  const [longitude, setLongitude] = useState('78.0');
  const [forecast, setForecast] = useState<ForecastPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(event?: FormEvent) {
    event?.preventDefault();
    const lat = Number(latitude);
    const lon = Number(longitude);
    if (!Number.isFinite(lat) || lat < 8 || lat > 37 || !Number.isFinite(lon) || lon < 68 || lon > 97) {
      setError('Choose a point inside the supported domain: 8–37°N, 68–97°E.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setForecast(asRecord(await fetchOperationalForecast(lat, lon)));
    } catch (exception) {
      setForecast(null);
      setError(exception instanceof Error ? exception.message : 'Operational API unavailable.');
    } finally {
      setLoading(false);
    }
  }

  const sourceMeta = asRecord(forecast?.source_metadata);
  const gfs = asRecord(forecast?.raw_gfs);
  const gfsData = asRecord(gfs.data);
  const gfsRain = gfs.status === 'LIVE' ? gfsData.gfs_daily_rain_mm : null;
  const explanation = asRecord(forecast?.anvaya_explanation);
  const warning = asRecord(forecast?.warnings);
  const regimeName = asString(forecast?.regime, 'UNAVAILABLE');
  const regimeProbabilities = asRecord(forecast?.regime_probability);
  const regimeConfidence = typeof regimeProbabilities[regimeName] === 'number' ? regimeProbabilities[regimeName] : null;
  const targetWindow = asRecord(forecast?.target_window);
  const topContributions = asRecords(explanation.top_feature_contributions);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-muted-foreground">LIVE_OPERATIONAL_MODE · point forecast, not a district mean or scientific benchmark.</p>
        </div>
        <span className="rounded-full border px-3 py-1 text-xs font-semibold">LIVE DATA · SOURCE STATUS SHOWN</span>
      </div>

      <Card>
        <CardContent className="pt-4">
          <form onSubmit={load} className="grid grid-cols-1 items-end gap-3 sm:grid-cols-[1fr_1fr_auto]">
            <label className="space-y-1 text-xs font-medium text-muted-foreground">
              Latitude (8–37°N)
              <input aria-label="Latitude" type="number" min="8" max="37" step="0.01" value={latitude} onChange={(event) => setLatitude(event.target.value)} className="h-10 w-full rounded-md border bg-background px-3 text-sm text-foreground" />
            </label>
            <label className="space-y-1 text-xs font-medium text-muted-foreground">
              Longitude (68–97°E)
              <input aria-label="Longitude" type="number" min="68" max="97" step="0.01" value={longitude} onChange={(event) => setLongitude(event.target.value)} className="h-10 w-full rounded-md border bg-background px-3 text-sm text-foreground" />
            </label>
            <button type="submit" disabled={loading} className="h-10 rounded-md bg-primary px-5 font-medium text-primary-foreground disabled:opacity-50">
              {loading ? 'Loading sources…' : 'Get live forecast'}
            </button>
          </form>
          {error && <p role="alert" className="mt-3 text-sm text-red-600">{error}</p>}
        </CardContent>
      </Card>

      {forecast && <>
        <div className="grid gap-3 sm:grid-cols-3">
          <Metric title="Identified regime" value={regimeName} status={regimeName !== 'UNAVAILABLE' ? 'MODEL POSTERIOR' : 'UNAVAILABLE'} />
          <Metric title="Regime posterior" value={probabilityText(regimeConfidence)} status="MODEL OUTPUT" />
          <Metric title="Forecast stability" value={probabilityText(asRecord(forecast.forecast_confidence).score)} status={asRecord(forecast.forecast_confidence).calibrated ? 'CALIBRATED' : 'NOT CALIBRATED'} />
        </div>

        <Card>
          <CardHeader><CardTitle>24-hour rainfall comparison · mm</CardTitle></CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-3">
              <ComparisonTile label="RAW GFS" value={gfsRain} status={sourceStatus(gfs)} />
              <ComparisonTile label="GLOBAL ML" value={forecast.global_ml_rain_mm} status={sourceStatus(sourceMeta.anvaya)} />
              <ComparisonTile label="ANVAYA" value={forecast.anvaya_corrected} status={sourceStatus(sourceMeta.anvaya)} highlight />
            </div>
            <div className="mt-4 flex flex-wrap gap-x-8 gap-y-2 border-t border-slate-800 pt-3 font-mono text-xs text-slate-400">
              <span>RAW → GLOBAL ML <strong className="ml-2 text-slate-100">{differenceText(forecast.global_ml_rain_mm, gfsRain)}</strong></span>
              <span>GLOBAL ML → ANVAYA <strong className="ml-2 text-cyan-200">{numberText(forecast.anvaya_minus_global_mm, 1, ' mm')}</strong></span>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 lg:grid-cols-3">
          <Card>
            <CardHeader><CardTitle>Probabilities & warning rule</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <Probability label="HEAVY · ≥64.5 mm" value={forecast.heavy_probability} />
              <Probability label="VERY HEAVY · ≥115.6 mm" value={forecast.very_heavy_probability} />
              <Probability label="EXTREME · ≥204.5 mm" value={forecast.extreme_probability} />
              <div className="border-t pt-3">
                <div className="text-xs text-muted-foreground">Operational decision rule</div>
                <div className="mt-1 text-2xl font-bold">{asString(forecast.warning_level, 'UNAVAILABLE')}</div>
                <p className="mt-1 text-xs text-muted-foreground">{asString(warning.disclaimer, 'Not an official IMD warning; thresholds are configurable and not calibrated as a public warning protocol.')}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Predicted rainfall regime</CardTitle></CardHeader>
            <CardContent>
              <div className="text-xl font-semibold">{regimeName}</div>
              <div className="mt-3 space-y-2">
                {Object.entries(regimeProbabilities).length ? Object.entries(regimeProbabilities).map(([name, probability]) => (
                  <div key={name} className="flex justify-between gap-4 text-sm"><span>{name}</span><span className="tabular-nums">{probabilityText(probability)}</span></div>
                )) : <p className="text-sm text-muted-foreground">Regime probabilities unavailable.</p>}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Model-based explanation</CardTitle></CardHeader>
            <CardContent>
              {explanation.status === 'LIVE' ? <>
                <p className="text-sm">Top local predictors for the ANVAYA − Global ML correction (not causal):</p>
                <ul className="mt-3 space-y-2 text-sm">
                  {topContributions.map((item, index) => (
                    <li key={asString(item.feature, String(index))} className="flex justify-between gap-3 border-b pb-2">
                      <span>{asString(item.physical_driver, asString(item.feature))}</span>
                      <span className="shrink-0 tabular-nums">{numberText(item.correction_contribution_mm, 2, ' mm')}</span>
                    </li>
                  ))}
                </ul>
              </> : <p className="text-sm text-muted-foreground">Explanation unavailable: {asString(explanation.error, 'the frozen model did not return supported local attributions.')}</p>}
              <p className="mt-3 text-xs text-muted-foreground">IMERG is observation context only and is not used as a future predictor. Live inputs are not benchmark-validated.</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader><CardTitle>Multi-source status</CardTitle></CardHeader>
          <CardContent className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
            <Source label="GFS" item={sourceMeta.gfs} />
            <Source label="GEFS" item={sourceMeta.gefs} />
            <Source label="ECMWF" item={sourceMeta.ecmwf} />
            <Source label="IMERG context" item={forecast.imerg_current} />
            <div className="rounded-md border border-slate-800 p-3"><div className="text-slate-500">MODEL AGREEMENT</div><div className="mt-1 font-semibold">{typeof asRecord(forecast.ensemble_consensus).independent_model_families === 'number' ? `${asRecord(forecast.ensemble_consensus).independent_model_families} independent families` : 'UNAVAILABLE'}</div><div className="mt-1 text-slate-500">Agreement is withheld unless valid times and accumulation windows match.</div></div>
            <div className="rounded-md border border-slate-800 p-3"><div className="text-slate-500">FORECAST SPREAD</div><div className="mt-1 font-semibold">{numberText(forecast.gefs_spread, 1, ' mm')}</div><div className="mt-1 text-slate-500">GEFS member spread; not calibrated confidence.</div></div>
            <div className="sm:col-span-2 lg:col-span-4 text-muted-foreground">Run: {asString(forecast.forecast_reference_time)} · Valid: {asString(targetWindow.start)} to {asString(targetWindow.end)} · Updated: {asString(forecast.timestamp)}</div>
          </CardContent>
        </Card>
      </>}
    </section>
  );
}

function Metric({ title, value, status }: { title: string; value: string; status: string }) {
  return <Card><CardContent className="pt-4">
    <div className="flex justify-between gap-2 text-xs text-muted-foreground"><span>{title}</span><span>{status}</span></div>
    <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
  </CardContent></Card>;
}

function Probability({ label, value }: { label: string; value: unknown }) {
  const probability = typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
  return <div className="space-y-1.5">
    <div className="flex justify-between gap-3 text-xs"><span>{label}</span><span className="font-semibold tabular-nums">{probabilityText(value)}</span></div>
    <div className="h-1.5 overflow-hidden rounded bg-slate-800"><div className="h-full bg-cyan-400" style={{ width: probability === null ? '0%' : `${probability * 100}%` }} /></div>
  </div>;
}

function ComparisonTile({ label, value, status, highlight = false }: { label: string; value: unknown; status: string; highlight?: boolean }) {
  return <div className={`rounded-md border p-4 ${highlight ? 'border-cyan-900 bg-cyan-950/30' : 'border-slate-800 bg-slate-950/50'}`}>
    <div className="flex items-center justify-between gap-2 font-mono text-[10px] tracking-wider text-slate-400"><span>{label}</span><span>{status}</span></div>
    <div className="mt-2 text-3xl font-semibold tabular-nums text-white">{numberText(value, 1)}<span className="ml-1 text-sm font-normal text-slate-500">mm</span></div>
  </div>;
}

function differenceText(left: unknown, right: unknown) {
  if (typeof left !== 'number' || typeof right !== 'number') return '—';
  return `${left - right >= 0 ? '+' : ''}${(left - right).toFixed(1)} mm`;
}

function Source({ label, item }: { label: string; item: unknown }) {
  const source = asRecord(item);
  const metadata = asRecord(source.metadata);
  return <div className="rounded-md border p-3">
    <div className="flex justify-between"><span className="font-semibold">{label}</span><span>{sourceStatus(source)}</span></div>
    <div className="mt-1 text-muted-foreground">{asString(source.source, asString(metadata.source, 'No source metadata returned'))}</div>
    {typeof source.valid_time === 'string' && <div className="mt-1 text-muted-foreground">Valid: {source.valid_time}</div>}
    {typeof source.error === 'string' && <div className="mt-1 break-words text-red-600">{source.error}</div>}
  </div>;
}
