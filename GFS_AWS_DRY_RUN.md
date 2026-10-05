# ANVAYA GFS archive dry-run

Checked: 2026-10-04 (Asia/Calcutta)  
Mode: metadata/index inspection only; no GFS forecast payload downloaded.  
Plan source: `config/feature_registry.yaml` and
`config/data_sources.yaml`; destination follows the existing raw-data layout.

## Proposed source and file set

The official NOAA AWS bucket is `s3://noaa-gfs-bdp-pds` in `us-east-1`. The
UCAR GFS d084001 record points to this bucket as its continuously updated AWS
copy and describes 0.25° GRIB2 grids with daily runs at 00Z, 06Z, 12Z, and 18Z.
The AWS key pattern for the current ANVAYA `lead_hours=24` default is:

```text
https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{YYYYMMDD}/{CC}/atmos/gfs.t{CC}z.pgrb2.0p25.f024
https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{YYYYMMDD}/{CC}/atmos/gfs.t{CC}z.pgrb2.0p25.f024.idx
```

For the 2020 historical portion, the UCAR GDEX archive uses this dated file
path and corresponding THREDDS NetCDF Subset Service endpoint (replace `f024`
with the selected lead):

```text
https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/{YYYY}/{YYYYMMDD}/gfs.0p25.{YYYYMMDD}{CC}.f024.grib2
https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001/{YYYY}/{YYYYMMDD}/gfs.0p25.{YYYYMMDD}{CC}.f024.grib2
```

The second endpoint supports a requested variable and bounding-box subset;
the first is the full-file route. Confirm each requested date/cycle/lead in
the UCAR catalog before retrieval.

For each date from June 1 through September 30 in 2021–2025, and each cycle
`00`, `06`, `12`, `18`, the candidate set is one GRIB2 forecast object plus
its small `.idx` sidecar: 2,440 cycle/file pairs in total. The `.idx` sidecar
contains byte offsets for individual GRIB fields and enables HTTP range reads.

The S3 `gfs.2020` prefix returned zero keys in the read-only listing. Therefore
the AWS plan cannot supply the requested 2020 monsoon period. UCAR GDEX d084001
lists 2020 in its historical archive; use its subset request for those 488
cycle/date combinations, after checking exact file availability. Do not assume
the AWS bucket contains 2020 just because the UCAR collection does.

## Feature-driven field selection

These are the only raw GFS fields in the current feature registry. The other
registered features are derived, static, calendar/forecast metadata, or
antecedent rainfall from an observation source.

| ANVAYA feature | GRIB2 short name and level | Candidate f024 record |
|---|---|---|
| `gfs_rain_mm` | `APCP`, surface, 0–1 day accumulation | Daily accumulated precipitation |
| `gfs_mslp` | `PRMSL`, mean sea level | Mean sea-level pressure |
| `gfs_t2m` | `TMP`, 2 m above ground | 2 m temperature |
| `gfs_rh2m` | `RH`, 2 m above ground | 2 m relative humidity |
| `gfs_u10` | `UGRD`, 10 m above ground | 10 m zonal wind |
| `gfs_v10` | `VGRD`, 10 m above ground | 10 m meridional wind |
| `gfs_u850` | `UGRD`, 850 mb | 850 mb zonal wind |
| `gfs_v850` | `VGRD`, 850 mb | 850 mb meridional wind |
| `gfs_cape` | `CAPE`, surface | Convective available potential energy |
| `gfs_precipitable_water` | `PWAT`, entire atmosphere | Precipitable water |

The mappings and f024 records were confirmed in representative 2021 and 2025
`.idx` sidecars. `gfs_rain_mm` uses the `APCP:surface:0-1 day acc fcst` record
for this size estimate. The selected byte ranges summed to approximately 8.3
MB for the sampled 2021 f024 file.

## Size estimate

Observed sizes from S3 object listings:

| Sample | Full GRIB2 object | `.idx` sidecar |
|---|---:|---:|
| 2021-06-01 00Z f024 | 552,866,865 bytes | 41,275 bytes |
| 2025-06-01 00Z f024 | 546,198,259 bytes | 41,253 bytes |

Using the two-file mean as a rough per-cycle estimate:

| Retrieval plan | Approximate transfer |
|---|---:|
| 2021–2025 full GRIB2 files, 2,440 cycles | 1.34 TB, plus about 0.10 GB of index sidecars |
| 2021–2025 selected ten GRIB messages by byte range | about 20.3 GB, plus about 0.10 GB of index sidecars |
| 2020 full-file equivalent, 488 cycles | about 268 GB; this is a rough proxy, not an AWS size measurement |
| Entire 2020–2025 full-file equivalent, 2,928 cycles | about 1.61 TB; 2020 must come from UCAR or another verified archive |

Selected-field estimate uses the observed 8.3 MB sum of ten f024 messages from
one 2021 cycle multiplied by 2,440 cycles. Actual transfer will vary by date,
forecast cycle, lead, GRIB packing, and missing/changed records. These GRIB
messages still contain global fields. The estimate excludes retries and any
GDEX-subset packaging differences.

## Filtering before transfer

| Filter | NOAA AWS S3 objects | UCAR GDEX subset route |
|---|---|---|
| Variables | Yes at the GRIB-message level: fetch `.idx`, then use HTTP byte-range GETs for the selected GRIB records. | Yes; GDEX supports variable selection. |
| India bounding box (8–37°N, 68–97°E) | No S3 server-side geographic crop. The selected messages are global; crop after decoding locally. | Yes; GDEX supports geographic subsetting before delivery. |

The existing `scripts/download_data.py` downloads complete objects; it does not
yet issue byte-range requests. The lower-volume AWS plan therefore needs a
range-aware retrieval step. For 2020, and when geographic filtering should
happen before transfer, use a GDEX subset request. Destination for raw GRIB2
staging is `data/raw/gfs/aws_grib2/`; keep the existing NetCDF/Parquet ingestion
path separate until a GRIB2-to-ANVAYA schema adapter is validated.

## Important timing assumption before ingestion

This dry-run uses forecast hour `f024` because the current dataset builder
defaults to `lead_hours=24`. It does not settle the observation-window join.
IMD daily rainfall in ANVAYA ends at 03Z, while f024 from the four standard
cycles is valid at 00Z, 06Z, 12Z, or 18Z. The rainfall accumulation interval
also needs to match the IMD window. Choose and test the cycle/lead/accumulation
alignment before converting these candidate files into training rows.

## Result

The dry-run is complete. No bulk download was started. The proposed raw staging
destination is `data/raw/gfs/aws_grib2/`. Estimates and cycle counts above are
planning figures; verify each day/cycle key and the relevant index records
before any retrieval begins.
