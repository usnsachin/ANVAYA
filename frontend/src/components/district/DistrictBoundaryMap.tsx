'use client';

export type DistrictFeature = {
  type: 'Feature';
  geometry: { type: 'Polygon' | 'MultiPolygon'; coordinates: number[][][] | number[][][][] };
  properties?: Record<string, unknown>;
};

export type FeatureCollection = { type: 'FeatureCollection'; features: DistrictFeature[] };

const EXTENT = { west: 68.0, east: 98.0, south: 6.5, north: 37.5 };
const WIDTH = 720;
const HEIGHT = 620;

function project(position: number[]) {
  const x = ((position[0] - EXTENT.west) / (EXTENT.east - EXTENT.west)) * WIDTH;
  const y = ((EXTENT.north - position[1]) / (EXTENT.north - EXTENT.south)) * HEIGHT;
  return [x, y] as const;
}

function ringPath(ring: number[][]) {
  if (!ring.length) return '';
  return `${ring.map((position, index) => {
    const [x, y] = project(position);
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ')} Z`;
}

function geometryPath(feature: DistrictFeature) {
  const polygons = feature.geometry.type === 'Polygon'
    ? [feature.geometry.coordinates as number[][][]]
    : feature.geometry.coordinates as number[][][][];
  return polygons.map((polygon) => polygon.map(ringPath).join(' ')).join(' ');
}

function propertyText(properties: Record<string, unknown>, name: string) {
  const value = properties[name];
  return typeof value === 'string' || typeof value === 'number' ? String(value) : '';
}

export function regimeColor(value: string) {
  const regime = value.toLocaleLowerCase();
  if (regime.includes('orographic')) return '#1769AA';
  if (regime.includes('convective')) return '#D64545';
  if (regime.includes('monsoon')) return '#2E8B57';
  if (regime.includes('break')) return '#9270A8';
  return '#68859A';
}

function fillColor(properties: Record<string, unknown>, mapStyle: 'standard' | 'dark') {
  const status = propertyText(properties, 'forecast_status');
  const layer = propertyText(properties, 'forecast_layer');
  const warning = propertyText(properties, 'warning_level');
  const value = properties.forecast_value;
  const neutral = mapStyle === 'dark' ? '#29475B' : '#E4EBEF';
  if (status !== 'LIVE' || value == null) return neutral;
  if (layer === 'risk') {
    return ({ RED: '#D34F9A', ORANGE: '#D64545', YELLOW: '#E9B949', GREEN: '#2E8B57' } as Record<string, string>)[warning] || neutral;
  }
  if (layer === 'regime' && typeof value === 'string') return regimeColor(value);
  if (typeof value !== 'number' || !Number.isFinite(value)) return neutral;
  if (layer === 'correction') return value > 0 ? '#D64545' : value < 0 ? '#1769AA' : '#79909F';
  return value >= 204.5 ? '#D34F9A' : value >= 115.6 ? '#D64545' : value >= 64.5 ? '#E9B949' : '#2E8B57';
}

export default function DistrictBoundaryMap({
  collection, selectedDistrictId, onSelect, selectedCoordinate, mapStyle = 'standard', layer = 'risk', showBoundaries = true,
}: {
  collection: FeatureCollection | null;
  selectedDistrictId?: string;
  onSelect?: (id: string) => void;
  selectedCoordinate?: { latitude: number; longitude: number } | null;
  mapStyle?: 'standard' | 'dark';
  layer?: string;
  showBoundaries?: boolean;
}) {
  if (!collection?.features?.length) return <div className="flex min-h-72 items-center justify-center rounded-md border border-dashed border-[#c8d5de] bg-white px-5 text-center text-sm text-[#667b8a]">District boundary reference layer is unavailable.</div>;
  const dark = mapStyle === 'dark';
  const [markerX, markerY] = selectedCoordinate ? project([selectedCoordinate.longitude, selectedCoordinate.latitude]) : [null, null];
  const markerInView = markerX !== null && markerY !== null && markerX >= 0 && markerX <= WIDTH && markerY >= 0 && markerY <= HEIGHT;

  return <div className={`overflow-hidden rounded-md border ${dark ? 'border-[#33546a] bg-[#12324A]' : 'border-[#d5e0e6] bg-[#edf3f6]'}`}>
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="mx-auto max-h-[72vh] min-h-[330px] w-full" role="group" aria-label={`India district map showing ${layer}; queried districts are colored and neutral boundaries are reference only`}>
      <rect width={WIDTH} height={HEIGHT} fill={dark ? '#12324A' : '#edf3f6'} />
      {collection.features.map((feature) => {
        const properties = feature.properties || {};
        const id = propertyText(properties, 'anvaya_district_id');
        const selected = id === selectedDistrictId;
        const name = propertyText(properties, 'district') || 'District';
        const state = propertyText(properties, 'state_name') || 'India';
        const label = `${name}, ${state}`;
        return <path
          key={id || label}
          d={geometryPath(feature)}
          fill={fillColor(properties, mapStyle)}
          fillRule="evenodd"
          stroke={selected ? '#1769AA' : showBoundaries ? dark ? '#527184' : '#a8b9c4' : 'transparent'}
          strokeWidth={selected ? 2 : 0.55}
          className={onSelect ? 'cursor-pointer transition-[fill,filter] hover:brightness-95 focus-visible:outline-none' : undefined}
          onClick={onSelect && id ? () => onSelect(id) : undefined}
          onKeyDown={onSelect && id ? (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(id); } } : undefined}
          tabIndex={onSelect && id ? 0 : undefined}
          role={onSelect && id ? 'button' : undefined}
          aria-label={onSelect && id ? `Select ${label}${properties.forecast_status === 'LIVE' ? `, forecast ${String(properties.forecast_value)}` : ', reference boundary only'}` : undefined}
        ><title>{label}{properties.forecast_status === 'LIVE' && properties.forecast_value != null ? ` · ${propertyText(properties, 'forecast_layer')}: ${propertyText(properties, 'forecast_value')}` : ''}</title></path>;
      })}
      {markerInView && <g aria-label="Selected resolved location">
        <circle cx={markerX!} cy={markerY!} r="13" fill="#ffffff" fillOpacity="0.85" />
        <circle cx={markerX!} cy={markerY!} r="7" fill="#D64545" stroke="#ffffff" strokeWidth="2.5" />
      </g>}
      <text x="18" y="30" fill={dark ? '#b8cbd7' : '#536a7a'} fontSize="12" fontFamily="sans-serif" letterSpacing="1.5">INDIA · DISTRICT BOUNDARIES</text>
    </svg>
  </div>;
}
