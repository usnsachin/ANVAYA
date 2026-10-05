'use client';

import { useEffect, useState } from 'react';
import { fetchHealth } from '@/lib/api';

type Health = { status?: string; model_status?: string; data_mode?: string; model_version?: string };

export default function StatusBar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let active = true;
    fetchHealth().then((response: Health) => {
      if (active) setHealth(response);
    }).catch(() => {
      if (active) setHealth(null);
    }).finally(() => {
      if (active) setChecked(true);
    });
    return () => { active = false; };
  }, []);

  const apiOnline = health?.status === 'ok' || health?.status === 'healthy';
  const modelReady = health?.model_status === 'ready';

  return <div className="flex max-w-full items-center gap-2 overflow-x-auto font-mono text-[9px] uppercase tracking-wider sm:gap-3">
    <Status label="API" value={!checked ? 'CHECKING' : apiOnline ? 'ONLINE' : 'OFFLINE'} tone={!checked ? 'muted' : apiOnline ? 'good' : 'bad'} />
    <Status label="SCIENCE" value="FROZEN 2024" tone="good" />
    <span className="hidden max-w-40 truncate text-slate-500 lg:block" title={health?.model_version || 'Model metadata unavailable'}>{checked && modelReady ? health?.model_version || 'Model ready' : 'MODEL STATUS UNKNOWN'}</span>
  </div>;
}

function Status({ label, value, tone }: { label: string; value: string; tone: 'good' | 'bad' | 'muted' }) {
  const color = tone === 'good' ? 'border-emerald-900 text-emerald-300' : tone === 'bad' ? 'border-rose-900 text-rose-300' : 'border-slate-800 text-slate-500';
  return <span className={`whitespace-nowrap rounded border px-2 py-1 ${color}`}><span className="text-slate-500">{label}</span> <span>{value}</span></span>;
}
