"""
Regime Label definitions and meteorological weak-supervision labeling rules for ANVAYA.

IMPORTANT METEOROLOGICAL DISCLAIMER:
These rules represent heuristic weak-supervision proxy labels developed for research prototype
post-processing. They are NOT official India Meteorological Department (IMD) synoptic classifications.
"""
from typing import Dict, Any

REGIME_LABELS = [
    'ACTIVE_MONSOON',
    'BREAK_MONSOON',
    'LOW_DEPRESSION',
    'COASTAL_RAINFALL',
    'OROGRAPHIC_RAINFALL',
    'WESTERN_DISTURBANCE',
    'OTHER'
]

REGIME_METADATA = {
    'ACTIVE_MONSOON': {
        'name': 'Active Monsoon',
        'code': 'ACTIVE_MONSOON',
        'description': 'Enhanced monsoon phase characterized by a robust cross-equatorial low-level jet, deep tropospheric moisture, and widespread precipitation over central and peninsular India.',
        'color': '#2563eb'
    },
    'BREAK_MONSOON': {
        'name': 'Break Monsoon',
        'code': 'BREAK_MONSOON',
        'description': 'Suppressed monsoon phase with the monsoon trough shifting north to the Himalayan foothills, dry northerly/westerly flow, and reduced rainfall over central India.',
        'color': '#dc2626'
    },
    'LOW_DEPRESSION': {
        'name': 'Low / Depression',
        'code': 'LOW_DEPRESSION',
        'description': 'Synoptic-scale cyclonic vortex or low-pressure system (typically originating in the Bay of Bengal or Arabian Sea) driving intense organized convergence and localized heavy precipitation.',
        'color': '#7c3aed'
    },
    'COASTAL_RAINFALL': {
        'name': 'Coastal Rainfall',
        'code': 'COASTAL_RAINFALL',
        'description': 'Offshore trough and strong southwesterly maritime inflow driving heavy coastal precipitation along the Konkan, Goa, and Malabar coasts.',
        'color': '#0891b2'
    },
    'OROGRAPHIC_RAINFALL': {
        'name': 'Orographic Rainfall',
        'code': 'OROGRAPHIC_RAINFALL',
        'description': 'Mechanically forced ascent of moist airmasses against steep topography (Western Ghats windward escarpment, Meghalaya plateau, and sub-Himalayan ridges).',
        'color': '#059669'
    },
    'WESTERN_DISTURBANCE': {
        'name': 'Western Disturbance',
        'code': 'WESTERN_DISTURBANCE',
        'description': 'Extra-tropical upper-tropospheric westerly trough propagating into northwest India, inducing cloudiness and rainfall in northern mountain and foothill zones.',
        'color': '#d97706'
    },
    'OTHER': {
        'name': 'Other / Transitional',
        'code': 'OTHER',
        'description': 'Transitional or mixed atmospheric state with weak synoptic forcing and no single dominant regime pattern.',
        'color': '#6b7280'
    }
}

def assign_weak_regime_label(row: Dict[str, Any]) -> str:
    """
    Transparent meteorological proxy heuristic to assign a weak regime label
    from physical atmosphere variables.
    """
    lat = float(row.get('latitude', 20.0))
    lon = float(row.get('longitude', 78.0))
    mslp = float(row.get('gfs_mslp', 1008.0))
    rh = float(row.get('gfs_rh2m', 70.0))
    u850 = float(row.get('gfs_u850', 5.0))
    v850 = float(row.get('gfs_v850', 2.0))
    pw = float(row.get('gfs_precipitable_water', 40.0))
    
    # 1. Cyclonic Low / Depression: Significant synoptic pressure departure
    if mslp < 998.0:
        return 'LOW_DEPRESSION'
        
    # 2. Western Disturbance: Northern latitudes with northerly/northwesterly cold advection
    if lat >= 28.0 and v850 < -2.0:
        return 'WESTERN_DISTURBANCE'
        
    # 3. Coastal Rainfall: West coast peninsular corridor with onshore southwesterlies
    is_west_coast = (lon <= 75.5 and 8.0 <= lat <= 20.0)
    if is_west_coast and u850 >= 9.0 and rh >= 78.0:
        return 'COASTAL_RAINFALL'
        
    # 4. Orographic Rainfall: Steep terrain zones (Western Ghats or Northeast foothills)
    is_northeast = (lat >= 24.5 and lon >= 88.0)
    is_ghats_crest = (73.5 <= lon <= 75.5 and 10.0 <= lat <= 19.5)
    if (is_northeast or is_ghats_crest) and (u850 >= 7.0 or v850 >= 5.0) and pw >= 48.0:
        return 'OROGRAPHIC_RAINFALL'
        
    # 5. Active Monsoon: Broadscale strong low-level westerly monsoon jet + high moisture
    if u850 >= 6.0 and pw >= 50.0 and rh >= 72.0:
        return 'ACTIVE_MONSOON'
        
    # 6. Break Monsoon: Weak low-level flow or dry conditions over central India
    if (15.0 <= lat <= 26.0) and (u850 <= 4.0 or rh <= 60.0):
        return 'BREAK_MONSOON'
        
    return 'OTHER'
