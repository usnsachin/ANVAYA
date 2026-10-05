'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import OperationalForecastPanel from '@/components/operational/OperationalForecastPanel';

type Scenario = { rain: number; westerly: number; humidity: number; cape: number; elevation: number; slope: number };
const WAYANAD: Scenario = { rain: 92, westerly: 19, humidity: 91, cape: 1400, elevation: 900, slope: 21 };

function sigmoid(value: number) { return 1 / (1 + Math.exp(-value)); }
function clamp(value: number, min: number, max: number) { return Math.max(min, Math.min(max, value)); }
function pct(value: number) { return `${(value * 100).toFixed(0)}%`; }
function confidenceText(value: number) { return `${value >= 0.8 ? 'HIGH' : value >= 0.65 ? 'MEDIUM' : 'LOW'} · ${pct(value)}`; }

function simulateScenario(input: Scenario) {
  const global = Math.max(0, input.rain * 0.93);
  const moisture = Math.max(0, input.humidity - 60) * 0.17 + input.cape / 250;
  const orographic = Math.max(0, input.westerly) * 1.15 + Math.max(0, input.elevation - 250) * 0.018 + input.slope * 0.45;
  const correction = clamp(moisture + orographic, 0, 85);
  const anvaya = global + correction;
  const heavy = sigmoid((anvaya - 64.5) / 24);
  const veryHeavy = sigmoid((anvaya - 115.6) / 22);
  const extreme = sigmoid((anvaya - 204.5) / 27);
  const confidence = clamp(0.55 + input.humidity / 100 * 0.15 + Math.min(Math.max(input.westerly, 0) / 25, 1) * 0.12 + Math.min(input.cape / 2000, 1) * 0.1 + Math.min(input.slope / 30, 1) * 0.08, 0, 0.95);
  const regime = input.westerly >= 8 && input.humidity >= 78 && input.elevation >= 650 && input.slope >= 12
    ? 'OROGRAPHIC RAINFALL'
    : input.cape >= 1200 && input.humidity >= 70 ? 'CONVECTIVE RAINFALL' : 'MONSOON RAINFALL';
  const severity = heavy >= 0.85 && veryHeavy >= 0.5 ? 'HIGH SCENARIO RISK' : heavy >= 0.5 ? 'ELEVATED SCENARIO RISK' : 'LOW SCENARIO RISK';
  return { global, anvaya, correction: anvaya - global, heavy, veryHeavy, extreme, confidence, regime, severity, moisture, orographic };
}

export default function ForecastPage() {
  const [scenario, setScenario] = useState(WAYANAD);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);

  function update(key: keyof Scenario, value: number) {
    setScenario((current) => ({ ...current, [key]: value }));
  }

  return <div className="space-y-7 pb-10">
    <header className="flex flex-wrap items-end justify-between gap-3">
      <div><p className="font-mono text-[10px] uppercase tracking-[0.2em] text-amber-300">DEMO · Scenario lab</p><h1 className="mt-1 text-2xl font-semibold tracking-tight">Interactive rainfall scenario</h1></div>
      <button type="button" onClick={() => setScenario(WAYANAD)} className="rounded border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-900">Reset Wayanad preset</button>
    </header>

    <section className="overflow-hidden rounded-lg border border-amber-900/70 bg-slate-900/60">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-amber-900/60 bg-amber-950/30 px-4 py-3">
        <div><p className="font-mono text-[10px] uppercase tracking-[0.2em] text-amber-300">Preconfigured demonstration case</p><h2 className="mt-1 text-lg font-semibold text-white">WAYANAD EXTREME OROGRAPHIC RAINFALL</h2></div>
        <span className="rounded border border-amber-800 px-2 py-1 font-mono text-[9px] font-bold tracking-wider text-amber-200">DEMO · SYNTHETIC</span>
      </div>
      <div className="grid gap-5 p-4 xl:grid-cols-[minmax(280px,0.82fr)_minmax(0,1.2fr)]">
        <div>
          <div className="rounded border border-amber-900/60 bg-amber-950/20 p-3 text-xs leading-relaxed text-amber-100/80">
            Illustrative what-if visualization only—not a real historical event, live forecast, calibrated probability, or Phase 3F model inference. The current operational API accepts location queries, not these overrides; terrain slope is not a registered Phase 3F feature. No slider value is sent to a provider or model.
          </div>
          <div className="mt-4 space-y-3">
            <ScenarioSlider label="Rainfall baseline" unit="mm / 24h" min={0} max={250} step={1} value={scenario.rain} onChange={(value) => update('rain', value)} />
            <ScenarioSlider label="850 hPa westerly wind · eastward positive" unit="m/s" min={-10} max={30} step={1} value={scenario.westerly} onChange={(value) => update('westerly', value)} />
            <ScenarioSlider label="Relative humidity" unit="%" min={20} max={100} step={1} value={scenario.humidity} onChange={(value) => update('humidity', value)} />
            <ScenarioSlider label="CAPE" unit="J/kg" min={0} max={3000} step={50} value={scenario.cape} onChange={(value) => update('cape', value)} />
            <ScenarioSlider label="Terrain elevation proxy" unit="m" min={0} max={3000} step={25} value={scenario.elevation} onChange={(value) => update('elevation', value)} />
            <ScenarioSlider label="Windward terrain slope · demo-only" unit="°" min={0} max={45} step={1} value={scenario.slope} onChange={(value) => update('slope', value)} />
          </div>
        </div>

        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <DemoMetric label="Scenario regime" value={result.regime} detail="Rule-based label · demo only" />
            <DemoMetric label="Scenario confidence" value={confidenceText(result.confidence)} detail="Heuristic score · not model confidence" />
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-mono text-[10px] uppercase tracking-widest text-slate-400">Illustrative forecast path · mm / 24h</h3><span className="font-mono text-[9px] text-amber-300">SCRIPTED DEMO CALCULATOR</span></div>
            <div className="mt-4 grid gap-2 sm:grid-cols-3">
              <RainMetric label="RAW NWP" value={scenario.rain} />
              <RainMetric label="GLOBAL ML · illustrative" value={result.global} />
              <RainMetric label="ANVAYA PATH · illustrative" value={result.anvaya} highlight />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 font-mono text-xs text-slate-400"><span>RAW</span><span>→</span><span>GLOBAL ML</span><span className="text-slate-200">{(result.global - scenario.rain >= 0 ? '+' : '') + (result.global - scenario.rain).toFixed(1)} mm</span><span>→</span><span>ANVAYA</span><span className="text-cyan-200">{result.correction >= 0 ? '+' : ''}{result.correction.toFixed(1)} mm</span></div>
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 border-t border-slate-800 pt-3 text-[10px] text-slate-500"><span>Demo moisture contribution <strong className="text-slate-300">+{result.moisture.toFixed(1)} mm</strong></span><span>Demo terrain-flow contribution <strong className="text-slate-300">+{result.orographic.toFixed(1)} mm</strong></span></div>
          </div>

          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
            <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
              <h3 className="font-mono text-[10px] uppercase tracking-widest text-slate-400">Nested threshold probabilities · illustrative</h3>
              <DemoProbability label="HEAVY · ≥64.5 mm" value={result.heavy} />
              <DemoProbability label="VERY HEAVY · ≥115.6 mm" value={result.veryHeavy} />
              <DemoProbability label="EXTREME · ≥204.5 mm" value={result.extreme} />
              <p className="mt-3 text-[10px] text-slate-500">Heuristic sigmoid display; not calibrated, trained, or suitable for warning decisions.</p>
            </div>
            <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-4">
              <p className="font-mono text-[10px] uppercase tracking-widest text-amber-300">Scenario severity</p>
              <p className="mt-2 text-lg font-semibold text-amber-100">{result.severity}</p>
              <p className="mt-2 text-xs text-amber-100/60">Not an official alert or observed district impact.</p>
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="font-mono text-[10px] uppercase tracking-widest text-slate-400">Scenario footprint · Wayanad focus</h3><span className="font-mono text-[9px] text-slate-500">REFERENCE GEOGRAPHY ONLY</span></div>
            <div className="mt-3 flex flex-wrap gap-2">{['Wayanad · scenario focus', 'Kozhikode · not computed', 'Kannur · not computed', 'Malappuram · not computed'].map((district) => <span key={district} className="rounded border border-slate-800 px-2 py-1 text-[10px] text-slate-300">{district}</span>)}</div>
            <p className="mt-2 text-[10px] text-slate-500">These are geographic labels only; no district forecast is generated by this demo.</p>
            <Link href="/map" className="mt-3 inline-flex text-xs font-medium text-cyan-200 hover:text-cyan-100">Query Wayanad with live operational sources →</Link>
          </div>

          <details className="rounded border border-slate-800 px-4 py-3 text-xs text-slate-400">
            <summary className="cursor-pointer font-medium text-slate-300">How this illustrative output is generated</summary>
            <p className="mt-2">The demo calculator applies explicit monotonic scenario weights to the humidity, CAPE, westerly-flow, elevation, and slope controls. Threshold values are transformed with simple sigmoid curves. These coefficients are presentation-only and are not learned or scientifically calibrated.</p>
          </details>
        </div>
      </div>
    </section>

    <div className="border-t border-slate-800 pt-6">
      <OperationalForecastPanel title="LIVE · Location-based operational forecast" />
    </div>
  </div>;
}

function ScenarioSlider({ label, unit, min, max, step, value, onChange }: { label: string; unit: string; min: number; max: number; step: number; value: number; onChange: (value: number) => void }) {
  return <label className="block">
    <span className="flex items-baseline justify-between gap-2 text-xs"><span className="text-slate-300">{label}</span><span className="font-mono tabular-nums text-cyan-200">{value} <span className="text-slate-500">{unit}</span></span></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} className="mt-2 h-1.5 w-full cursor-pointer accent-cyan-300" aria-label={label} />
  </label>;
}

function DemoMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4"><p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{label}</p><p className="mt-2 text-lg font-semibold text-white">{value}</p><p className="mt-1 text-[10px] text-amber-300/80">{detail}</p></div>;
}

function RainMetric({ label, value, highlight = false }: { label: string; value: number; highlight?: boolean }) {
  return <div className={`rounded border p-3 ${highlight ? 'border-cyan-900 bg-cyan-950/20' : 'border-slate-800'}`}><p className="font-mono text-[9px] tracking-wider text-slate-500">{label}</p><p className="mt-2 text-2xl font-semibold tabular-nums text-white">{value.toFixed(1)}<span className="ml-1 text-xs text-slate-500">mm</span></p></div>;
}

function DemoProbability({ label, value }: { label: string; value: number }) {
  return <div className="mt-3"><div className="flex justify-between gap-2 text-xs"><span>{label}</span><span className="font-mono tabular-nums text-white">{pct(value)}</span></div><div className="mt-1.5 h-1.5 overflow-hidden rounded bg-slate-800"><div className="h-full bg-amber-300" style={{ width: `${value * 100}%` }} /></div></div>;
}
