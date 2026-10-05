'use client';
import { useEffect, useState } from 'react';
import { fetchHealth } from '@/lib/api';

export default function StatusBadge() {
  const [status, setStatus] = useState<'loading' | 'connected' | 'disconnected'>('loading');

  useEffect(() => {
    fetchHealth()
      .then(() => setStatus('connected'))
      .catch(() => setStatus('disconnected'));
  }, []);

  return (
    <div className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border bg-zinc-50 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400">
      <span className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-emerald-500' : status === 'disconnected' ? 'bg-red-500' : 'bg-zinc-400 animate-pulse'}`} />
      {status === 'connected' ? 'Connected' : status === 'disconnected' ? 'Disconnected' : 'Checking...'}
    </div>
  );
}
