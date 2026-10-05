"""Offline tests for strict UCAR GFS subset mapping and request planning."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.ucar_gfs_subset import (
    FEATURES,
    REQUEST_GROUPS,
    UCARGFSSSubsetError,
    UCARGFSSSubsetProvider,
    _decode_numeric_cf_time_variables,
    _metadata_grid_records,
)


class UCARGFSSSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = UCARGFSSSubsetProvider(root=ROOT)

    def test_registry_features_match_all_ten_metadata_mappings(self) -> None:
        import yaml

        registry = yaml.safe_load((ROOT / "config/feature_registry.yaml").read_text(encoding="utf-8"))
        registered = {
            item["feature_name"] for item in registry["features"]
            if item["feature_name"].startswith("gfs_")
        } - {"gfs_daily_rain_mm"}
        self.assertEqual(set(FEATURES), registered)
        self.assertEqual(len(FEATURES), 10)

    def test_metadata_parser_retains_grib_identity_and_rejects_missing_fields(self) -> None:
        xml = b'''<gridDataset><grid name="u-component_of_wind_isobaric" desc="u wind">
        <attribute name="Grib2_Parameter" value="0 2 2" />
        <attribute name="Grib2_Level_Type" value="100" />
        <attribute name="units" value="m/s" /></grid></gridDataset>'''
        _, records = _metadata_grid_records(xml)
        self.assertEqual(records["u-component_of_wind_isobaric"]["Grib2_Parameter"], "0 2 2")
        self.assertNotIn(FEATURES["gfs_v850"].variable, records)

    def test_ncss_request_groups_select_only_verified_compatible_levels(self) -> None:
        plans = self.provider._query_groups()
        self.assertEqual(len(plans), 6)
        params = {tuple(features): query for features, query in plans}
        self.assertEqual(params[("gfs_t2m",)]["vertCoord"], 2.0)
        self.assertEqual(params[("gfs_rh2m",)]["vertCoord"], 2.0)
        self.assertEqual(params[("gfs_u10", "gfs_v10")]["vertCoord"], 10.0)
        self.assertEqual(params[("gfs_u850", "gfs_v850")]["vertCoord"], 85000.0)
        self.assertNotIn("vertCoord", params[("gfs_mslp", "gfs_cape", "gfs_precipitable_water")])
        self.assertTrue(all(query["accept"] == "netcdf3" for query in params.values()))
        self.assertTrue(all("addLatLon" not in query for query in params.values()))
        self.assertIn(("gfs_apcp_interval_mm",), params)
        self.assertNotIn(("gfs_t2m", "gfs_rh2m"), params)
        self.assertNotIn("vertCoord", params[("gfs_apcp_interval_mm",)])

    def test_single_case_url_uses_exact_archive_file_and_positive_longitudes(self) -> None:
        init = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
        url = self.provider._dataset_url(init, 24)
        self.assertEqual(
            url,
            "https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001/2024/20240715/gfs.0p25.2024071500.f024.grib2",
        )
        bbox_queries = [params for _, params in self.provider._query_groups()]
        self.assertTrue(all(p["west"] == 68.0 and p["east"] == 97.0 for p in bbox_queries))

    def test_provider_is_separate_and_has_no_fallback_path(self) -> None:
        from ml.data.providers import GFSProvider

        self.assertIsNot(UCARGFSSSubsetProvider, GFSProvider)
        self.assertEqual(self.provider.mode, "historical")
        self.assertEqual(self.provider.source_name, "UCAR_GFS_0p25_THREDDS_NCSS")
        self.assertEqual(self.provider.manifest_path, ROOT / "data/raw/manifests/gfs_test_subset.json")

    def test_raw_apcp_is_never_exposed_as_daily_or_legacy_rain(self) -> None:
        point = self.provider.get_features(20.0, 78.0, lead_hours=24)
        self.assertIn("gfs_apcp_interval_mm", point)
        self.assertNotIn("gfs_daily_rain_mm", point)
        self.assertNotIn("gfs_rain_mm", point)

    def test_numeric_cf_time_fallback_decodes_uppercase_hour_units_and_bounds(self) -> None:
        import numpy as np
        import xarray as xr

        units = "Hour since 2017-12-10T00:00:00Z"
        dataset = xr.Dataset(
            data_vars={
                "time_bounds": (("time", "bounds"), [[0.0, 6.0]], {"units": units, "calendar": "standard"}),
                "sample": (("time",), [1.0]),
            },
            coords={
                "time": xr.DataArray(
                    [6.0], dims=("time",),
                    attrs={"units": units, "calendar": "standard", "standard_name": "time", "bounds": "time_bounds"},
                ),
                "bounds": [0, 1],
            },
        )
        decoded = _decode_numeric_cf_time_variables(dataset)
        self.assertEqual(
            np.datetime_as_string(decoded["time"].values[0], unit="s"),
            "2017-12-10T06:00:00",
        )
        self.assertEqual(
            np.datetime_as_string(decoded["time_bounds"].values[0, 1], unit="s"),
            "2017-12-10T06:00:00",
        )
        self.assertEqual(decoded["time"].attrs["source_time_units"], units)


if __name__ == "__main__":
    unittest.main(verbosity=2)
