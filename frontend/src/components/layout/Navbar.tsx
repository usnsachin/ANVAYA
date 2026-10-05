'use client';

import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Bell, ChartNoAxesCombined, House, Map as MapIcon } from 'lucide-react';

const navLinks = [
  { name: 'HOME', href: '/', Icon: House },
  { name: 'MAP', href: '/map', Icon: MapIcon },
  { name: 'ALERTS', href: '/alerts', Icon: Bell },
  { name: 'INSIGHTS', href: '/insights', Icon: ChartNoAxesCombined },
];

function activeRoute(pathname: string, href: string) {
  if (href === '/') return pathname === '/';
  if (href === '/insights') return pathname === '/insights' || pathname === '/verification';
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Navbar() {
  const pathname = usePathname();
  return <>
    <header className="sticky top-0 z-40 border-b border-[#d9e2e8] bg-white/95 shadow-[0_2px_12px_rgba(18,50,74,0.04)] backdrop-blur">
      <div className="mx-auto flex h-[68px] max-w-[1320px] items-center justify-between gap-5 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex min-w-0 shrink-0 items-center gap-2.5" aria-label="ANVAYA home">
          <span className="grid h-11 w-12 shrink-0 place-items-center overflow-hidden rounded-md bg-[#12324A] px-0.5">
            <Image src="/branding/anvaya-logo.png" alt="ANVAYA emblem" width={56} height={48} priority className="h-10 w-11 object-contain" />
          </span>
          <span className="min-w-0"><span className="block text-[15px] font-bold tracking-[0.11em] text-[#12324A]">ANVAYA</span><span className="hidden text-[10px] text-[#667b8a] sm:block">Regime-Aware Rainfall Intelligence</span></span>
        </Link>

        <nav aria-label="Primary" className="hidden h-full items-center gap-1 md:flex">
          {navLinks.map(({ name, href }) => {
            const active = activeRoute(pathname, href);
            return <Link key={href} href={href} aria-current={active ? 'page' : undefined} className={`inline-flex h-full items-center border-b-2 px-3 text-[11px] font-semibold tracking-[0.09em] transition-colors ${active ? 'border-[#1769AA] text-[#1769AA]' : 'border-transparent text-[#536a7a] hover:text-[#12324A]'}`}>{name}</Link>;
          })}
        </nav>

        <Link href="/forecast" aria-current={pathname === '/forecast' ? 'page' : undefined} className="shrink-0 rounded-md border border-[#d5a95f] bg-[#fffaf0] px-3 py-2 text-[10px] font-semibold text-[#79551d] hover:bg-[#fff3d9]">Scenario lab <span className="ml-1 rounded bg-[#f3e1bc] px-1 py-0.5 text-[8px] tracking-wide">DEMO</span></Link>
      </div>
    </header>

    <nav aria-label="Primary mobile" className="fixed inset-x-0 bottom-0 z-50 grid grid-cols-4 border-t border-[#d9e2e8] bg-white/97 pb-[env(safe-area-inset-bottom)] shadow-[0_-4px_16px_rgba(18,50,74,0.08)] md:hidden">
      {navLinks.map(({ name, href, Icon }) => {
        const active = activeRoute(pathname, href);
        return <Link key={href} href={href} aria-current={active ? 'page' : undefined} className={`flex min-h-[60px] flex-col items-center justify-center gap-1 text-[9px] font-semibold tracking-[0.07em] ${active ? 'text-[#1769AA]' : 'text-[#667b8a]'}`}>
          <Icon aria-hidden="true" className="h-[18px] w-[18px]" strokeWidth={active ? 2.4 : 1.8} />{name}
        </Link>;
      })}
    </nav>
  </>;
}
