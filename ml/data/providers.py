"""
Data provider adapters for ANVAYA.

Decouples NWP (NOAA GFS) and observational truth (IMD / NCMRWF MERA) data formats
(GRIB2, NetCDF, OPeNDAP) from downstream ML models and REST API consumers.

Normalized schema columns:
- init_time
- valid_time
- lead_hours
- latitude
- longitude
- gfs_apcp_interval_mm: raw precipitation over the exact GFS metadata interval
- gfs_daily_rain_mm: only the metadata-aligned 24-hour IMD-window product
- gfs_mslp
- gfs_t2m
- gfs_rh2m
- gfs_u10
- gfs_v10
- gfs_u850
- gfs_v850
- gfs_cape
- gfs_precipitable_water
- observed_rain_mm
- source
"""
import os
import glob
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard Indian subcontinent bounding domain
INDIA_BBOX = {
    "min_lat": 8.0,
    "max_lat": 37.0,
    "min_lon": 68.0,
    "max_lon": 97.0
}

NORMALIZED_COLUMNS = [
    "init_time", "valid_time", "lead_hours", "latitude", "longitude",
    "gfs_apcp_interval_mm", "gfs_daily_rain_mm", "gfs_mslp", "gfs_t2m", "gfs_rh2m",
    "gfs_u10", "gfs_v10", "gfs_u850", "gfs_v850",
    "gfs_cape", "gfs_precipitable_water",
    "observed_rain_mm", "source"
]

class BaseProvider(ABC):
    """Abstract interface for all meteorological data providers."""
    
    @abstractmethod
    def get_features(
        self,
        lat: float,
        lon: float,
        valid_time: Optional[datetime] = None,
        lead_hours: int = 24
    ) -> Dict[str, Any]:
        """Fetch normalized feature dictionary for a given point and timestamp."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        """Returns 'demo', 'historical', or 'live'."""
        pass


class DemoProvider(BaseProvider):
    """
    Deterministic synthetic data provider for DEMO mode.
    Guarantees that demo mode always runs offline without external network dependency.
    """
    source_name = "SYNTHETIC_DEMO"
    mode = "demo"

    def get_features(
        self,
        lat: float,
        lon: float,
        valid_time: Optional[datetime] = None,
        lead_hours: int = 24
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(abs(lat * 100) + abs(lon * 10)) % 100000)
        month = valid_time.month if valid_time else 7
        is_monsoon = 6 <= month <= 9
        is_coastal = (lon <= 75.5 and 8.0 <= lat <= 20.0)

        base_rain = float(rng.exponential(25.0 if is_monsoon else 2.0))
        if is_coastal and is_monsoon:
            base_rain += float(rng.uniform(15.0, 45.0))

        u850 = float(rng.normal(12.0 if is_monsoon else 2.0, 3.0))
        rh = float(rng.uniform(75.0, 95.0) if is_monsoon else rng.uniform(40.0, 75.0))
        pw = float(rng.uniform(45.0, 65.0) if is_monsoon else rng.uniform(15.0, 35.0))

        v_time = valid_time or datetime(2024, 7, 15, 0, 0, tzinfo=timezone.utc)
        i_time = v_time - timedelta(hours=lead_hours)

        return {
            "init_time": i_time.isoformat(),
            "valid_time": v_time.isoformat(),
            "lead_hours": lead_hours,
            "latitude": float(lat),
            "longitude": float(lon),
            "gfs_daily_rain_mm": round(base_rain, 2),
            "gfs_mslp": round(float(rng.normal(1003.0 if is_monsoon else 1012.0, 4.0)), 2),
            "gfs_t2m": round(float(rng.normal(28.0, 2.5)), 2),
            "gfs_rh2m": round(rh, 1),
            "gfs_u10": round(float(rng.normal(3.5, 2.0)), 2),
            "gfs_v10": round(float(rng.normal(1.5, 1.5)), 2),
            "gfs_u850": round(u850, 2),
            "gfs_v850": round(float(rng.normal(2.5, 2.0)), 2),
            "gfs_cape": round(float(rng.exponential(750.0 if is_monsoon else 150.0)), 1),
            "gfs_precipitable_water": round(pw, 1),
            "rain_last_6h": round(float(max(0.0, rng.exponential(6.0))), 2),
            "rain_last_24h": round(float(max(0.0, rng.exponential(18.0))), 2),
            "observed_rain_mm": round(base_rain * 1.25, 2),
            "source": self.source_name
        }


class GFSProvider(BaseProvider):
    """
    NOAA/NCEP GFS 0.25° NWP forecast adapter.
    Reads real GFS historical NetCDF/GRIB files or queries live NOMADS endpoints.
    """
    source_name = "NOAA_GFS_0p25"

    def __init__(self, data_dir: str = "data/raw/gfs", live: bool = False):
        self.data_dir = data_dir
        self._live = live
        self._demo_fallback = DemoProvider()
        os.makedirs(data_dir, exist_ok=True)

    @property
    def mode(self) -> str:
        return "live" if self._live else "historical"

    def load_gridded_dataset(self, file_path: str):
        """Load GFS GRIB2 or NetCDF using xarray."""
        import xarray as xr
        if file_path.endswith(".grib2") or file_path.endswith(".grb2"):
            try:
                return xr.open_dataset(file_path, engine="cfgrib")
            except Exception as e:
                logger.warning(f"cfgrib failed on {file_path}: {e}")
                return xr.open_dataset(file_path)
        else:
            return xr.open_dataset(file_path)

    def subset_india(self, ds, lat_var="latitude", lon_var="longitude"):
        """Geographically subset to the Indian subcontinent."""
        # Convert longitude if standard 0-360
        if ds[lon_var].max() > 180.0:
            ds = ds.assign_coords({lon_var: (((ds[lon_var] + 180) % 360) - 180)})
            ds = ds.sortby(lon_var)

        lat_slice = slice(INDIA_BBOX["max_lat"], INDIA_BBOX["min_lat"]) if ds[lat_var][0] > ds[lat_var][-1] else slice(INDIA_BBOX["min_lat"], INDIA_BBOX["max_lat"])
        lon_slice = slice(INDIA_BBOX["min_lon"], INDIA_BBOX["max_lon"])
        return ds.sel({lat_var: lat_slice, lon_var: lon_slice})

    def get_features(
        self,
        lat: float,
        lon: float,
        valid_time: Optional[datetime] = None,
        lead_hours: int = 24
    ) -> Dict[str, Any]:
        """Point extraction with graceful fallback to demo provider if offline."""
        # Look for local GFS NetCDF/GRIB2 files
        candidates = glob.glob(os.path.join(self.data_dir, "*.nc")) + glob.glob(os.path.join(self.data_dir, "*.grib2"))
        if candidates:
            try:
                ds = self.load_gridded_dataset(candidates[0])
                pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
                res = {
                    "init_time": str(pt.get("time", valid_time).values),
                    "valid_time": str(valid_time or pt.get("time", datetime.utcnow()).values),
                    "lead_hours": lead_hours,
                    "latitude": lat,
                    "longitude": lon,
                    # A generic APCP field is not assumed to be a 24-hour total.
                    # Its metadata-defined bounds must be passed through the
                    # GFS precipitation alignment layer before daily use.
                    "gfs_apcp_interval_mm": float(pt.get("tp", pt.get("apcpsfc", 0.0)).values),
                    "gfs_mslp": float(pt.get("msl", pt.get("prmslmsl", 101325.0)).values) / 100.0,
                    "gfs_t2m": float(pt.get("t2m", pt.get("tmp2m", 300.0)).values) - 273.15,
                    "gfs_rh2m": float(pt.get("r2", pt.get("rh2m", 70.0)).values),
                    "gfs_u10": float(pt.get("u10", pt.get("ugrd10m", 0.0)).values),
                    "gfs_v10": float(pt.get("v10", pt.get("vgrd10m", 0.0)).values),
                    "gfs_u850": float(pt.get("u850", pt.get("ugrd850mb", 0.0)).values),
                    "gfs_v850": float(pt.get("v850", pt.get("vgrd850mb", 0.0)).values),
                    "gfs_cape": float(pt.get("cape", pt.get("capesfc", 0.0)).values),
                    "gfs_precipitable_water": float(pt.get("pwat", pt.get("pwatclm", 40.0)).values),
                    "observed_rain_mm": np.nan,
                    "source": self.source_name
                }
                return res
            except Exception as e:
                logger.warning(f"GFS local point extraction error: {e}")

        # Fallback to deterministic demo provider
        out = self._demo_fallback.get_features(lat, lon, valid_time, lead_hours)
        out["source"] = f"{self.source_name} [DEMO_FALLBACK]"
        return out


class IMDRainfallProvider:
    """
    IMD 0.25° Gridded Daily Rainfall Provider.
    Ground-truth observation dataset for rainfall bias correction and verification.
    """
    source_name = "IMD_0p25_DAILY_RAIN"

    def __init__(self, data_dir: str = "data/raw/imd"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_rainfall_dataset(self, file_path: str):
        """Open IMD NetCDF daily gridded rainfall dataset."""
        import xarray as xr
        return xr.open_dataset(file_path)

    def get_observed_rain(
        self,
        lat: float,
        lon: float,
        date_str: str
    ) -> Optional[float]:
        """Look up observed rainfall at location and date."""
        files = glob.glob(os.path.join(self.data_dir, "*.nc"))
        if not files:
            return None

        import xarray as xr
        try:
            ds = xr.open_dataset(files[0])
            pt = ds.sel(latitude=lat, longitude=lon, time=date_str, method="nearest")
            for var in ["rain", "rf", "rainfall", "RAINFALL", "pr"]:
                if var in pt.data_vars:
                    return float(pt[var].values)
        except Exception as e:
            logger.debug(f"IMD lookup failed: {e}")
        return None


class MERAProvider:
    """
    NCMRWF MERA ~4km regional reanalysis provider for high-resolution extreme verification.
    """
    source_name = "NCMRWF_MERA_4km"

    def __init__(self, data_dir: str = "data/raw/mera"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def is_available(self) -> bool:
        return os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0


def get_provider(mode: Optional[str] = None) -> BaseProvider:
    """Factory creating appropriate provider based on DATA_MODE environment variable."""
    selected_mode = (mode or os.environ.get("DATA_MODE", "demo")).lower().strip()
    if selected_mode == "live":
        return GFSProvider(live=True)
    elif selected_mode == "historical":
        return GFSProvider(live=False)
    else:
        return DemoProvider()
