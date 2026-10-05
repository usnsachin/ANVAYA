"""
Feature Engineering module for ANVAYA.
Extracts synoptic, dynamic, and thermodynamic derived features from raw meteorological fields.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

FEATURE_SCHEMA = [
    # Raw GFS inputs; APCP is admitted only after exact IMD-window aggregation.
    'gfs_daily_rain_mm', 'gfs_mslp', 'gfs_t2m', 'gfs_rh2m',
    'gfs_u10', 'gfs_v10', 'gfs_u850', 'gfs_v850',
    'gfs_cape', 'gfs_precipitable_water',
    # Recent observations
    # Daily RF25 supports only a previous completed 24h target, not a 6h field.
    'rain_last_24h',
    # Spatial coordinates
    'latitude', 'longitude',
    # Derived dynamical & thermodynamic features
    'wind_speed_10m', 'wind_dir_10m',
    'wind_speed_850', 'moisture_transport_850',
    'mslp_anomaly', 'lead_time_h',
    # Geographic & topological proxies (placeholders)
    'coast_distance_approx', 'terrain_elevation_proxy', 'orographic_flow_proxy',
    'month', 'day_of_year'
]

# Minimal feature set required for inference
MINIMUM_FEATURES = [
    'gfs_daily_rain_mm', 'gfs_mslp', 'gfs_t2m', 'gfs_rh2m',
    'gfs_u10', 'gfs_v10', 'gfs_u850', 'gfs_v850',
    'gfs_cape', 'gfs_precipitable_water',
    'rain_last_24h',
    'latitude', 'longitude'
]

def calculate_wind_speed(u: float, v: float) -> float:
    """Calculate horizontal wind speed in m/s."""
    return float(np.sqrt(u**2 + v**2))

def calculate_wind_direction(u: float, v: float) -> float:
    """Calculate meteorological wind direction (degrees from North)."""
    return float((np.degrees(np.arctan2(u, v)) + 360) % 360)

def calculate_mslp_anomaly(mslp: float, standard_mslp: float = 1013.25) -> float:
    """Calculate mean sea level pressure departure from standard atmosphere in hPa."""
    return float(mslp - standard_mslp)

def calculate_moisture_transport_proxy(wind_speed_850: float, precipitable_water: float) -> float:
    """Estimate low-level moisture transport proxy (m/s * kg/m^2)."""
    return float(wind_speed_850 * max(0.0, precipitable_water))

def calculate_orographic_flow_proxy(u850: float, v850: float, lat: float, lon: float) -> float:
    """
    Proxy for orographic uplift against Western Ghats and Himalayan slopes.
    For Western Ghats (lon ~73-77, lat 8-21): westerly zonal flow (u > 0) forces uplift.
    For Himalayas (lat > 27): southerly meridional flow (v > 0) forces uplift.
    """
    uplift = 0.0
    if 8.0 <= lat <= 21.0 and 72.0 <= lon <= 77.0:
        # Western Ghats windward exposure: positive westerly zonal wind
        uplift += max(0.0, u850) * 1.5
    elif lat >= 26.0:
        # Himalayan foothill exposure: southerly inflow
        uplift += max(0.0, v850) * 1.2
    return float(uplift)

def calculate_coast_distance_approx(lat: float, lon: float) -> float:
    """
    Approximate distance from Indian coastline in degrees.
    West Coast roughly around lon 73-76. East Coast roughly around lon 80-87.
    """
    # Rough distance to nearest oceanic boundary for peninsular India
    if lat < 24.0:
        dist_west = abs(lon - 73.0)
        dist_east = abs(lon - 83.0)
        return float(min(dist_west, dist_east))
    return float(abs(lat - 24.0) * 0.8 + 2.0)

def calculate_terrain_elevation_proxy(lat: float, lon: float) -> float:
    """
    Coarse topographical elevation proxy in meters.
    Western Ghats ridge (73-75E, 9-19N) ~ 600-1200m.
    Deccan plateau ~ 400-600m.
    Himalayan region (lat > 28) ~ 800-4000m.
    Indo-Gangetic plain ~ 50-200m.
    """
    if lat > 30.0:
        return float(1500.0 + (lat - 30.0) * 400.0)
    if 10.0 <= lat <= 20.0 and 73.5 <= lon <= 75.5:
        return 850.0  # Western Ghats ridge
    if 12.0 <= lat <= 22.0 and 74.0 <= lon <= 79.0:
        return 500.0  # Deccan plateau
    return 100.0      # Coastal / Gangetic plains default

def engineer_features(
    row: Dict[str, Any],
    lead_time_h: int = 24,
    valid_date: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Transform raw GFS and observation dictionary into full feature schema.
    """
    out = dict(row)

    # 10m Wind dynamics
    u10 = float(out.get('gfs_u10', 0.0))
    v10 = float(out.get('gfs_v10', 0.0))
    out['wind_speed_10m'] = calculate_wind_speed(u10, v10)
    out['wind_dir_10m'] = calculate_wind_direction(u10, v10)

    # 850 hPa Low-Level Jet dynamics
    u850 = float(out.get('gfs_u850', 0.0))
    v850 = float(out.get('gfs_v850', 0.0))
    wind_850 = calculate_wind_speed(u850, v850)
    out['wind_speed_850'] = wind_850

    # Moisture flux
    pw = float(out.get('gfs_precipitable_water', 0.0))
    out['moisture_transport_850'] = calculate_moisture_transport_proxy(wind_850, pw)

    # Pressure dynamics
    mslp = float(out.get('gfs_mslp', 1013.25))
    out['mslp_anomaly'] = calculate_mslp_anomaly(mslp)

    # Lead time
    out['lead_time_h'] = float(lead_time_h)

    # Spatial & Topological proxies
    lat = float(out.get('latitude', 20.0))
    lon = float(out.get('longitude', 78.0))
    out['coast_distance_approx'] = calculate_coast_distance_approx(lat, lon)
    out['terrain_elevation_proxy'] = calculate_terrain_elevation_proxy(lat, lon)
    out['orographic_flow_proxy'] = calculate_orographic_flow_proxy(u850, v850, lat, lon)

    # Temporal context
    if valid_date is not None:
        ts = pd.Timestamp(valid_date)
        out['month'] = float(ts.month)
        out['day_of_year'] = float(ts.day_of_year)
    else:
        out['month'] = float(out.get('month', 7.0))
        out['day_of_year'] = float(out.get('day_of_year', 196.0))

    return out

def dataframe_from_features(
    rows: List[Dict[str, Any]],
    lead_time_h: int = 24,
    valid_date: Optional[Any] = None
) -> pd.DataFrame:
    """Vectorized feature engineering across a list of raw records."""
    engineered = [engineer_features(r, lead_time_h, valid_date) for r in rows]
    return pd.DataFrame(engineered)
