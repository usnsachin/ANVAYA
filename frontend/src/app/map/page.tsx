import OperationalMapSurface from '@/components/operational/OperationalMapSurface';

export default function MapPage() {
  return <div className="space-y-4 pb-5">
    <header className="border-b border-[#d9e2e8] pb-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#1769AA]">MAP · GEOGRAPHY</p>
      <div className="mt-1 flex flex-wrap items-end justify-between gap-2">
        <div><h1 className="text-[25px] font-semibold tracking-tight text-[#12324A] sm:text-[30px]">Rainfall by location</h1><p className="mt-1 max-w-2xl text-sm text-[#536a7a]">Explore returned district outlooks and resolve one location at a time. Unqueried boundaries remain neutral.</p></div>
        <span className="rounded-sm border border-[#c9dbe8] bg-[#eff6fb] px-2.5 py-1.5 text-[9px] font-bold tracking-wide text-[#1769AA]">733 DISTRICTS · REFERENCE GEOMETRY</span>
      </div>
    </header>
    <OperationalMapSurface />
    <p className="text-[10px] leading-relaxed text-[#667b8a]">District colors are computed only after a forecast response is returned. A selected city or locality is shown as a point and is not promoted to an area-average forecast.</p>
  </div>;
}
