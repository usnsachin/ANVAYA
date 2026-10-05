'use client';

import { Card, CardContent } from '@/components/ui/card';

export type SystemMetrics = {
  rmse?: number | null;
  mae?: number | null;
  bias?: number | null;
  heavy?: {
    csi?: number | null;
    ets?: number | null;
    pod?: number | null;
    far?: number | null;
    fss_5x5?: number | null;
    threshold_mm?: number | null;
  };
  [key: string]: unknown;
};

export default function VerificationMetrics({
  metrics,
  period,
  comparison,
  source,
}: {
  metrics: SystemMetrics;
  period: string;
  comparison?: Record<string, SystemMetrics>;
  source?: string;
}) {
  const format = (value?: number | null, digits = 3) =>
    typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  const systems = [
    ['raw_gfs', 'Raw GFS'],
    ['global_ml', 'Global ML'],
    ['anvaya', 'ANVAYA'],
  ] as const;
  const heavy = (item?: SystemMetrics) => item?.heavy || {};

  return <div className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="text-sm text-muted-foreground">Period: {period}</div>
      <span className="rounded-full border border-emerald-900 bg-emerald-950/40 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-emerald-200">{source || 'PHASE3F_FROZEN'} · SCIENTIFIC_MODE</span>
    </div>

    <Card>
      <CardContent className="overflow-x-auto pt-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Held-out deterministic comparison</h3>
        <table className="w-full min-w-[760px] border-collapse text-left text-sm">
          <thead><tr className="border-b text-xs text-muted-foreground">
            <th className="py-2 pr-4">System</th><th className="px-3 py-2">RMSE (mm)</th><th className="px-3 py-2">MAE (mm)</th><th className="px-3 py-2">Bias (mm)</th><th className="px-3 py-2">Heavy CSI</th><th className="px-3 py-2">ETS</th><th className="px-3 py-2">POD</th><th className="px-3 py-2">FAR</th><th className="px-3 py-2">FSS 5×5</th>
          </tr></thead>
          <tbody>{systems.map(([key, label]) => {
            const item = comparison?.[key] || (key === 'anvaya' ? metrics : undefined);
            const event = heavy(item);
            return <tr key={key} className={`border-b last:border-0 ${key === 'anvaya' ? 'font-semibold' : ''}`}>
              <th className="py-3 pr-4">{label}</th>
              <td className="px-3 py-3 tabular-nums">{format(item?.rmse)}</td>
              <td className="px-3 py-3 tabular-nums">{format(item?.mae)}</td>
              <td className="px-3 py-3 tabular-nums">{format(item?.bias)}</td>
              <td className="px-3 py-3 tabular-nums">{format(event.csi)}</td>
              <td className="px-3 py-3 tabular-nums">{format(event.ets)}</td>
              <td className="px-3 py-3 tabular-nums">{format(event.pod)}</td>
              <td className="px-3 py-3 tabular-nums">{format(event.far)}</td>
              <td className="px-3 py-3 tabular-nums">{format(event.fss_5x5)}</td>
            </tr>;
          })}</tbody>
        </table>
        <p className="mt-3 text-xs text-muted-foreground">Heavy threshold: {format(heavy(comparison?.anvaya).threshold_mm, 1)} mm/day. FSS uses the frozen 5×5-grid-cell neighborhood score. These values are historical verification only.</p>
      </CardContent>
    </Card>

    <div className="grid gap-3 sm:grid-cols-3">
      {systems.map(([key, label]) => {
        const item = comparison?.[key];
        return <Card key={key}><CardContent className="pt-4">
          <div className="text-xs font-semibold uppercase text-muted-foreground">{label}</div>
          <div className="mt-2 text-sm">RMSE <strong className="tabular-nums">{format(item?.rmse)} mm</strong></div>
          <div className="text-sm">MAE <strong className="tabular-nums">{format(item?.mae)} mm</strong></div>
          <div className="text-sm">Heavy FSS <strong className="tabular-nums">{format(heavy(item).fss_5x5)}</strong></div>
        </CardContent></Card>;
      })}
    </div>
  </div>;
}
