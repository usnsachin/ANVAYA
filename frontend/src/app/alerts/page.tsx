'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowUpRight, MapPin } from 'lucide-react';
import Link from 'next/link';
import { useOperationalWorkspace, type DistrictForecast } from '@/components/operational/OperationalWorkspace';
import { displayNumber, displayPercent, displayText, warningCategory, utcLabel } from '@/components/operational/operational-format';

const FILTERS = ['ALL', 'EXTREME', 'HEAVY', 'MODERATE'] as const;

function categoryStyle(level: unknown) {
  const category = warningCategory(level);
  if (category === 'EXTREME') return { border: 'border-l-[#8E2028]', badge: 'border-[#d99da0] bg-[#faeeee] text-[#8E2028]' };
  if (category === 'HEAVY') return { border: 'border-l-[#D64545]', badge: 'border-[#efc0c0] bg-[#fff2f2] text-[#a93232]' };
  if (category === 'MODERATE') return { border: 'border-l-[#E97BA8]', badge: 'border-[#f0c5d6] bg-[#fff2f7] text-[#87425f]' };
  if (category === 'LOW') return { border: 'border-l-[#2E8B57]', badge: 'border-[#b9ddc8] bg-[#EAF5EE] text-[#246b43]' };
  return { border: 'border-l-[#9aaab5]', badge: 'border-[#d9e2e8] bg-[#f1f4f6] text-[#667b8a]' };
}

function actionFor(level: unknown) {
  const category = warningCategory(level);
  if (category === 'EXTREME') return 'Escalate local preparedness using established emergency procedures.';
  if (category === 'HEAVY') return 'Prepare for localized heavy rainfall and monitor local conditions.';
  if (category === 'MODERATE') return 'Review local readiness for possible heavy rainfall.';
  if (category === 'LOW') return 'This decision rule did not trigger a heavy-rain signal; continue normal monitoring.';
  return 'No action guidance is available because a forecast value was not returned.';
}

function severityOrder(level: unknown) {
  return ({ RED: 0, ORANGE: 1, YELLOW: 2, GREEN: 3 } as Record<string, number>)[String(level)] ?? 4;
}

export default function AlertsPage() {
  const router = useRouter();
  const { computedDistricts, selectDistrict } = useOperationalWorkspace();
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('ALL');
  const [query, setQuery] = useState('');
  const rows = useMemo(() => computedDistricts
    .filter((row) => {
      const district = row.district || {};
      const name = `${displayText(district.district, '')} ${displayText(district.state, '')}`.toLocaleLowerCase();
      const category = warningCategory(row.warning_level);
      return name.includes(query.trim().toLocaleLowerCase()) && (filter === 'ALL' || category === filter);
    })
    .sort((a, b) => severityOrder(a.warning_level) - severityOrder(b.warning_level)), [computedDistricts, filter, query]);

  return <div className="space-y-5 pb-5">
    <header className="border-b border-[#d9e2e8] pb-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#1769AA]">ALERTS · ACTION</p>
      <div className="mt-1 flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="text-[25px] font-semibold tracking-tight text-[#12324A] sm:text-[30px]">District decision feed</h1><p className="mt-1 max-w-2xl text-sm text-[#536a7a]">Only forecasts actually queried and returned in this browser session appear here.</p></div>
        <p className="text-xs text-[#667b8a]">{computedDistricts.length} district{computedDistricts.length === 1 ? '' : 's'} queried</p>
      </div>
    </header>

    <section className="rounded-md border border-[#d9e2e8] bg-white px-4 py-3.5 sm:px-5" aria-label="Alert policy note">
      <div className="flex items-start gap-3"><span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-sm bg-[#eff6fb] text-[#1769AA]"><MapPin aria-hidden="true" className="h-4 w-4" /></span><div><p className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#1769AA]">ANVAYA decision support</p><p className="mt-1 text-xs leading-relaxed text-[#536a7a]">Risk categories follow the configured ANVAYA decision rule. They are not official IMD warnings or government advisories, and this page is not an all-India bulletin.</p></div></div>
    </section>

    <div className="flex flex-col gap-3 rounded-md border border-[#d9e2e8] bg-white p-3 sm:flex-row sm:items-center sm:p-4">
      <label className="sr-only" htmlFor="alerts-search">Search queried districts</label>
      <input id="alerts-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search location or state" className="h-10 min-w-0 flex-1 rounded-sm border border-[#c6d3dd] bg-white px-3 text-sm text-[#12324A] outline-none placeholder:text-[#7b8b98] focus:border-[#1769AA]" />
      <div role="tablist" aria-label="Filter alert category" className="flex gap-1 overflow-x-auto">
        {FILTERS.map((item) => <button key={item} type="button" role="tab" aria-selected={filter === item} onClick={() => setFilter(item)} className={`min-h-9 shrink-0 rounded-sm px-3 text-[9px] font-bold tracking-[0.08em] ${filter === item ? 'bg-[#12324A] text-white' : 'border border-[#d9e2e8] text-[#536a7a] hover:bg-[#f1f7fb]'}`}>{item}</button>)}
      </div>
    </div>

    {!computedDistricts.length ? <div className="rounded-md border border-dashed border-[#c8d5de] bg-white px-5 py-12 text-center">
      <h2 className="text-base font-semibold text-[#12324A]">No district forecasts in this session</h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-[#667b8a]">The feed is not pre-populated with estimates. Query a district on the map to add its returned area-weighted forecast here.</p>
      <Link href="/map" className="mt-5 inline-flex min-h-10 items-center rounded-sm bg-[#1769AA] px-4 text-xs font-semibold text-white hover:bg-[#12598f]">Open rainfall map <ArrowUpRight aria-hidden="true" className="ml-2 h-4 w-4" /></Link>
    </div> : rows.length ? <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-3" aria-label="Queried district alerts">
      {rows.map((row) => <AlertCard key={rowKey(row)} row={row} onOpen={() => {
        const id = String(row.district?.district_id ?? '');
        if (id) selectDistrict(id);
        router.push('/map');
      }} />)}
    </section> : <div className="rounded-md border border-[#d9e2e8] bg-white px-4 py-9 text-center text-sm text-[#667b8a]">No queried locations match this category and search.</div>}

    <p className="text-[10px] leading-relaxed text-[#667b8a]">Feed items are browser-session snapshots, not persistent alerts. No unqueried district receives an inferred risk category.</p>
  </div>;
}

function AlertCard({ row, onOpen }: { row: DistrictForecast; onOpen: () => void }) {
  const district = row.district || {};
  const tone = categoryStyle(row.warning_level);
  const status = displayText(row.status, 'UNAVAILABLE');
  return <article className={`flex min-w-0 flex-col rounded-md border border-[#d9e2e8] border-l-[4px] bg-white p-4 shadow-[0_2px_10px_rgba(18,50,74,0.035)] ${tone.border}`}>
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0"><h2 className="truncate text-base font-semibold text-[#12324A]">{displayText(district.district)}</h2><p className="mt-0.5 truncate text-xs text-[#667b8a]">{displayText(district.state)}</p></div>
      <span className={`shrink-0 rounded-sm border px-2 py-1 text-[9px] font-bold tracking-wide ${tone.badge}`}>{warningCategory(row.warning_level)}</span>
    </div>
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 border-y border-[#e4ebef] py-3">
      <AlertMetric label="Rainfall · 24 h" value={displayNumber(row.anvaya_rain_mm, ' mm', 1)} />
      <AlertMetric label="Heavy probability" value={displayPercent(row.heavy_probability)} />
      <AlertMetric label="Weather regime" value={displayText(row.regime)} />
      <AlertMetric label="Forecast status" value={status} />
    </dl>
    <p className="mt-3 text-xs leading-relaxed text-[#536a7a]">{actionFor(row.warning_level)}</p>
    <p className="mt-2 text-[9px] text-[#667b8a]">Valid {utcLabel(row.valid_time)}</p>
    <button type="button" onClick={onOpen} className="mt-4 inline-flex min-h-9 items-center justify-center gap-1 rounded-sm border border-[#b7c9d5] px-3 text-xs font-semibold text-[#1769AA] hover:bg-[#f1f7fb]">View on map <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" /></button>
  </article>;
}

function AlertMetric({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0"><dt className="text-[10px] text-[#667b8a]">{label}</dt><dd className="mt-0.5 truncate text-sm font-semibold tabular-nums text-[#12324A]" title={value}>{value}</dd></div>;
}

function rowKey(row: DistrictForecast) {
  const district = row.district || {};
  return String(district.district_id ?? `${district.state}-${district.district}`);
}
