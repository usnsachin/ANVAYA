'use client';

import { useEffect, useMemo, useState } from 'react';
import { fetchVerification } from '@/lib/api';
import { asRecord } from '@/components/operational/OperationalWorkspace';

type MetricValue = number | null | undefined;
type TabId = 'overview' | 'regime' | 'extreme' | 'spatial';
type ChartDatum = { label: string; value: MetricValue; color: string };

const SYSTEMS = [
  { id: 'raw_gfs', label: 'RAW GFS', color: '#6C8799' },
  { id: 'global_ml', label: 'GLOBAL ML', color: '#5F9E75' },
  { id: 'anvaya', label: 'ANVAYA', color: '#1769AA' },
] as const;
const TABS: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'OVERVIEW' },
  { id: 'regime', label: 'BY REGIME' },
  { id: 'extreme', label: 'EXTREME RAIN' },
  { id: 'spatial', label: 'SPATIAL SKILL' },
];
const THRESHOLDS = ['heavy', 'very_heavy', 'extreme'] as const;

type BenchmarkData = {
  status?: string;
  period?: string;
  metadata?: Record<string, unknown>;
  comparison?: Record<string, Record<string, unknown>>;
  thresholds?: Record<string, Record<string, Record<string, unknown>>>;
  regime_stratified?: Record<string, Record<string, unknown>>;
  interpretation?: string;
};

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function format(value: unknown, digits = 3) {
  const number = numberValue(value);
  return number === null ? '—' : number.toFixed(digits);
}

function thresholdName(value: string) {
  return value.replaceAll('_', ' ').toLocaleUpperCase();
}

export default function InsightsPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>('overview');
  const [threshold, setThreshold] = useState<(typeof THRESHOLDS)[number]>('heavy');

  useEffect(() => {
    let active = true;
    fetchVerification()
      .then((response: BenchmarkData) => { if (active) setData(response); })
      .catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Scientific verification is unavailable.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const comparison = data?.comparison || {};
  const thresholdSystems = data?.thresholds || {};
  const regimeRows = useMemo(() => Object.entries(data?.regime_stratified || {}), [data?.regime_stratified]);
  const meta = data?.metadata || {};
  const benchmarkReady = data?.status === 'REFERENCE' && meta.benchmark_frozen === true;

  if (loading) return <div className="space-y-4" aria-live="polite"><div className="h-7 w-56 animate-pulse rounded bg-[#e6edf1]" /><div className="h-56 animate-pulse rounded-md bg-white" /><p className="text-xs text-[#667b8a]">Loading frozen forecast-performance data…</p></div>;
  if (!data || error) return <div role="alert" className="rounded-md border border-[#edc8c8] bg-white p-6"><p className="text-[10px] font-bold uppercase tracking-wide text-[#963232]">FORECAST PERFORMANCE · UNAVAILABLE</p><h1 className="mt-2 text-xl font-semibold text-[#12324A]">Scientific results could not be loaded</h1><p className="mt-2 text-sm text-[#536a7a]">{error || 'The verification endpoint returned no data. No demonstration metrics are substituted.'}</p></div>;

  return <div className="space-y-5 pb-5">
    <header className="border-b border-[#d9e2e8] pb-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#1769AA]">INSIGHTS · SCIENCE</p>
      <div className="mt-1 flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="text-[25px] font-semibold tracking-tight text-[#12324A] sm:text-[30px]">Forecast performance</h1><p className="mt-1 text-sm text-[#536a7a]">A read-only view of the existing held-out scientific evaluation.</p></div>
        <span className={`rounded-sm border px-2.5 py-1.5 text-[9px] font-bold tracking-wide ${benchmarkReady ? 'border-[#b9ddc8] bg-[#EAF5EE] text-[#246b43]' : 'border-[#d9e2e8] bg-[#f1f4f6] text-[#667b8a]'}`}>{benchmarkReady ? 'FROZEN · SCIENTIFIC MODE' : 'REFERENCE STATUS UNCONFIRMED'}</span>
      </div>
    </header>

    <section className="grid gap-3 sm:grid-cols-[1.2fr_0.8fr]">
      <div className="rounded-md border border-[#d9e2e8] bg-white p-4"><p className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#667b8a]">EVALUATION PERIOD</p><p className="mt-1 text-sm font-semibold text-[#12324A]">{data.period || 'Period not returned'}</p><p className="mt-1 text-xs text-[#667b8a]">{typeof meta.sample_size === 'number' ? `${meta.sample_size.toLocaleString()} held-out grid-day samples` : 'Sample count not returned'}{typeof meta.target_dates === 'number' ? ` · ${meta.target_dates.toLocaleString()} target dates` : ''}</p></div>
      <div className="rounded-md border border-[#d9e2e8] bg-white p-4"><p className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#667b8a]">SOURCE</p><p className="mt-1 text-sm font-semibold text-[#12324A]">{String(meta.source || 'Source metadata not returned')}</p><p className="mt-1 text-xs text-[#667b8a]">Live operational provider responses do not change these scores.</p></div>
    </section>

    <div role="tablist" aria-label="Scientific performance sections" className="flex gap-1 overflow-x-auto border-b border-[#d9e2e8]">
      {TABS.map((item) => <button key={item.id} id={`insights-tab-${item.id}`} type="button" role="tab" aria-selected={tab === item.id} aria-controls={`insights-panel-${item.id}`} onClick={() => setTab(item.id)} className={`min-h-10 shrink-0 border-b-2 px-3 text-[9px] font-bold tracking-[0.08em] ${tab === item.id ? 'border-[#1769AA] text-[#1769AA]' : 'border-transparent text-[#667b8a] hover:text-[#12324A]'}`}>{item.label}</button>)}
    </div>

    <section id={`insights-panel-${tab}`} role="tabpanel" aria-labelledby={`insights-tab-${tab}`} className="space-y-4">
      {tab === 'overview' && <>
        <div className="grid gap-4 lg:grid-cols-2">
          <MetricChart title="RMSE comparison" subtitle="mm · lower values indicate smaller error" data={comparisonSeries(comparison, 'rmse')} />
          <MetricChart title="MAE comparison" subtitle="mm · lower values indicate smaller absolute error" data={comparisonSeries(comparison, 'mae')} />
        </div>
        <ComparisonTable comparison={comparison} />
      </>}

      {tab === 'regime' && <>
        <GroupedRegimeChart title="Regime-wise RMSE" metric="rmse" note="mm · each bar uses the frozen regime-stratified result returned by the verification endpoint." rows={regimeRows} />
        <GroupedRegimeChart title="Regime-wise CSI" metric="csi" note="Threshold event CSI, stratified by the frozen regime labels." rows={regimeRows} />
        {!regimeRows.length && <EmptyChart>No regime-stratified results were returned for this frozen experiment.</EmptyChart>}
      </>}

      {tab === 'extreme' && <>
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-[#d9e2e8] bg-white px-4 py-3">
          <div><h2 className="text-sm font-semibold text-[#12324A]">Threshold performance</h2><p className="mt-0.5 text-[10px] text-[#667b8a]">Select a threshold to inspect event scores from the frozen results.</p></div>
          <label className="flex items-center gap-2 text-xs text-[#536a7a]">Rainfall threshold<select value={threshold} onChange={(event) => setThreshold(event.target.value as (typeof THRESHOLDS)[number])} className="h-9 rounded-sm border border-[#c6d3dd] bg-white px-2 text-xs font-semibold text-[#12324A]">{THRESHOLDS.map((item) => <option key={item} value={item}>{thresholdName(item)}</option>)}</select></label>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {(['csi', 'ets', 'pod', 'far'] as const).map((metric) => <MetricChart key={metric} title={`${metric.toLocaleUpperCase()} · ${thresholdName(threshold)}`} subtitle={metric === 'far' ? 'False alarm ratio · lower values indicate fewer false alarms' : `${metricName(metric)} · ${thresholdName(threshold)} events`} data={thresholdSeries(thresholdSystems, threshold, metric)} />)}
        </div>
        <GroupedThresholdChart thresholds={thresholdSystems} metric="csi" title="Threshold performance · CSI" note="Compares the existing CSI values for heavy, very-heavy and extreme thresholds." />
      </>}

      {tab === 'spatial' && <>
        <GroupedThresholdChart thresholds={thresholdSystems} metric="fss_5x5" title="Fractions Skill Score · 5×5 grid cells" note="Values are read directly from the frozen CSV-backed verification response. This panel does not recompute spatial skill." />
        <p className="rounded-md border border-[#d9e2e8] bg-white p-4 text-xs leading-relaxed text-[#536a7a]">FSS compares neighborhood fractions over the experiment&apos;s 5×5-cell window. Interpret it with the corresponding threshold and grid definition; it is not a point-forecast score.</p>
      </>}
    </section>

    <section className="rounded-md border border-[#d9e2e8] bg-[#eff6fb] p-4">
      <h2 className="text-xs font-semibold text-[#12324A]">Method and caveats</h2>
      <p className="mt-1 text-xs leading-relaxed text-[#536a7a]">Evaluation uses a held-out temporal test period. {data.interpretation || 'Interpretation is limited to the period and source reported above.'} The benchmark is frozen; operational forecasts, live APIs and session alerts are not included. No result here establishes performance outside the evaluated data and definitions.</p>
    </section>
  </div>;
}

function comparisonSeries(comparison: Record<string, Record<string, unknown>>, field: string): ChartDatum[] {
  return SYSTEMS.map((system) => ({ label: system.label, color: system.color, value: numberValue(comparison[system.id]?.[field]) }));
}

function thresholdSeries(thresholds: Record<string, Record<string, Record<string, unknown>>>, threshold: string, metric: string): ChartDatum[] {
  return SYSTEMS.map((system) => ({ label: system.label, color: system.color, value: numberValue(thresholds[system.id]?.[threshold]?.[metric]) }));
}

function metricName(metric: string) {
  return ({ csi: 'Critical Success Index', ets: 'Equitable Threat Score', pod: 'Probability of Detection', far: 'False Alarm Ratio', fss_5x5: 'Fractions Skill Score' } as Record<string, string>)[metric] || metric.toLocaleUpperCase();
}

function MetricChart({ title, subtitle, data }: { title: string; subtitle: string; data: ChartDatum[] }) {
  const valid = data.map((item) => item.value).filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const max = Math.max(...valid, 0);
  if (!valid.length) return <EmptyChart>{title}: no score was returned by the frozen verification endpoint.</EmptyChart>;
  return <article className="rounded-md border border-[#d9e2e8] bg-white p-4 sm:p-5">
    <header className="mb-4 border-b border-[#e7edf1] pb-3"><h2 className="text-sm font-semibold text-[#12324A]">{title}</h2><p className="mt-1 text-[10px] text-[#667b8a]">{subtitle}</p></header>
    <div className="space-y-4" role="img" aria-label={`${title}: ${data.map((item) => `${item.label} ${format(item.value)}`).join('; ')}`}>
      {data.map((item) => <div key={item.label} className="grid grid-cols-[80px_minmax(0,1fr)_52px] items-center gap-2">
        <span className="text-[10px] font-medium text-[#536a7a]">{item.label}</span>
        <div className="h-3 overflow-hidden rounded-sm bg-[#eef3f6]"><span className="block h-full rounded-sm" style={{ width: `${typeof item.value === 'number' ? Math.max(1, Math.min(100, max > 0 ? item.value / max * 100 : 0)) : 0}%`, backgroundColor: item.color }} /></div>
        <span className="text-right font-mono text-[10px] tabular-nums text-[#12324A]">{format(item.value)}</span>
      </div>)}
    </div>
  </article>;
}

function ComparisonTable({ comparison }: { comparison: Record<string, Record<string, unknown>> }) {
  return <div className="overflow-x-auto rounded-md border border-[#d9e2e8] bg-white">
    <table className="w-full min-w-[640px] text-left text-xs"><caption className="px-4 py-3 text-left text-sm font-semibold text-[#12324A]">Deterministic comparison · frozen endpoint values</caption>
      <thead className="bg-[#f1f6f8] text-[9px] uppercase tracking-wide text-[#667b8a]"><tr>{['System', 'RMSE · mm', 'MAE · mm', 'Bias · mm'].map((heading) => <th key={heading} className="px-4 py-2.5">{heading}</th>)}</tr></thead>
      <tbody>{SYSTEMS.map((system) => <tr key={system.id} className="border-t border-[#e7edf1]"><th className="px-4 py-3 font-semibold text-[#12324A]">{system.label}</th><td className="px-4 py-3 tabular-nums">{format(comparison[system.id]?.rmse)}</td><td className="px-4 py-3 tabular-nums">{format(comparison[system.id]?.mae)}</td><td className="px-4 py-3 tabular-nums">{format(comparison[system.id]?.bias)}</td></tr>)}</tbody>
    </table>
  </div>;
}

function GroupedRegimeChart({ title, metric, note, rows }: { title: string; metric: 'rmse' | 'csi'; note: string; rows: [string, Record<string, unknown>][] }) {
  const chartRows = rows.map(([regime, entry]) => {
    const systems = asRecord(entry.systems);
    return { regime, values: SYSTEMS.map((system) => ({ label: system.label, color: system.color, value: numberValue(systems[system.id] && asRecord(systems[system.id])[metric]) })) };
  });
  const valid = chartRows.flatMap((row) => row.values.map((item) => item.value)).filter((value): value is number => value !== null);
  const max = Math.max(...valid, 0);
  if (!chartRows.length || !valid.length) return <EmptyChart>{title}: no regime-wise values were returned.</EmptyChart>;
  return <article className="rounded-md border border-[#d9e2e8] bg-white p-4 sm:p-5">
    <header className="mb-4 border-b border-[#e7edf1] pb-3"><h2 className="text-sm font-semibold text-[#12324A]">{title}</h2><p className="mt-1 text-[10px] text-[#667b8a]">{note}</p></header>
    <ChartLegend />
    <div className="mt-4 space-y-5" role="img" aria-label={title}>
      {chartRows.map((row) => <div key={row.regime} className="grid gap-2 sm:grid-cols-[150px_minmax(0,1fr)] sm:items-center"><p className="text-[10px] font-semibold text-[#536a7a]">{row.regime.replaceAll('_', ' ')}</p><div className="space-y-1.5">{row.values.map((item) => <div key={item.label} className="grid grid-cols-[74px_minmax(0,1fr)_48px] items-center gap-2"><span className="text-[9px] text-[#667b8a]">{item.label}</span><div className="h-2.5 overflow-hidden rounded-sm bg-[#eef3f6]"><span className="block h-full" style={{ width: `${item.value === null ? 0 : Math.max(1, Math.min(100, max ? item.value / max * 100 : 0))}%`, backgroundColor: item.color }} /></div><span className="text-right font-mono text-[9px] tabular-nums text-[#12324A]">{format(item.value)}</span></div>)}</div></div>)}
    </div>
  </article>;
}

function GroupedThresholdChart({ thresholds, metric, title, note }: { thresholds: Record<string, Record<string, Record<string, unknown>>>; metric: string; title: string; note: string }) {
  const rows = THRESHOLDS.map((threshold) => ({
    threshold,
    values: SYSTEMS.map((system) => ({ label: system.label, color: system.color, value: numberValue(thresholds[system.id]?.[threshold]?.[metric]) })),
  }));
  const valid = rows.flatMap((row) => row.values.map((item) => item.value)).filter((value): value is number => value !== null);
  const max = Math.max(...valid, 0);
  if (!valid.length) return <EmptyChart>{title}: no values returned.</EmptyChart>;
  return <article className="rounded-md border border-[#d9e2e8] bg-white p-4 sm:p-5">
    <header className="mb-4 border-b border-[#e7edf1] pb-3"><h2 className="text-sm font-semibold text-[#12324A]">{title}</h2><p className="mt-1 text-[10px] text-[#667b8a]">{note}</p></header>
    <ChartLegend />
    <div className="mt-4 grid gap-4 sm:grid-cols-3" role="img" aria-label={title}>
      {rows.map((row) => <div key={row.threshold} className="min-w-0 rounded-sm bg-[#f6f9fa] p-3"><h3 className="mb-3 text-[10px] font-semibold text-[#12324A]">{thresholdName(row.threshold)}</h3><div className="space-y-2">{row.values.map((item) => <div key={item.label}><div className="mb-1 flex justify-between gap-2 text-[9px]"><span className="text-[#667b8a]">{item.label}</span><span className="font-mono tabular-nums text-[#12324A]">{format(item.value)}</span></div><div className="h-2 overflow-hidden rounded-sm bg-[#e7edf1]"><span className="block h-full" style={{ width: `${item.value === null ? 0 : Math.max(1, Math.min(100, max ? item.value / max * 100 : 0))}%`, backgroundColor: item.color }} /></div></div>)}</div></div>)}
    </div>
  </article>;
}

function ChartLegend() {
  return <div className="flex flex-wrap gap-x-4 gap-y-2" aria-label="Series legend">{SYSTEMS.map((system) => <span key={system.id} className="inline-flex items-center gap-1.5 text-[9px] text-[#536a7a]"><i className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: system.color }} />{system.label}</span>)}</div>;
}

function EmptyChart({ children }: { children: React.ReactNode }) {
  return <div className="rounded-md border border-dashed border-[#c8d5de] bg-white px-4 py-7 text-sm text-[#667b8a]">{children}</div>;
}
