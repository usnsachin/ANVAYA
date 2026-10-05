'use client';
import { useState } from 'react';

export default function DemoBanner() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200 px-4 py-2 text-xs text-center flex justify-between items-center z-50 relative">
      <span className="flex-1 font-medium">DEMO DATA — Values shown are synthetic and for development purposes only</span>
      <button onClick={() => setIsVisible(false)} className="hover:opacity-70 font-bold px-2">✕</button>
    </div>
  );
}
