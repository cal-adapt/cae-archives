# cadcat WRF + LOCA2 quality report

Generated 2026-09-23 19:56.

Inspected **630** dataset(s); recorded **1,961** finding(s).

## Severity

| level | count |
| --- | ---: |
| ERROR | 276 |
| WARN | 624 |
| INFO | 621 |
| OK | 440 |

## Findings by code

| level | code | count | example |
| --- | --- | ---: | --- |
| ERROR | `catalog.path.staging_prefix` | 104 | Store is served from a 'tmp' prefix rather than the canonical location. Data under a staging prefix carries no persistence guarantee, so anything pinned to this |
| ERROR | `catalog.path.variable_qualifier_dropped` | 104 | The store holds 'u10_earth' but the catalog indexes it as 'u10'. The '_earth' qualifier means earth-relative rather than grid-relative; WRF's native u10/v10 are |
| ERROR | `crs.spatial_ref.missing` | 40 | LOCA2 stores should carry a spatial_ref coordinate holding the WGS84 definition. |
| ERROR | `attr.var.long_name.missing` | 16 | Required variable attribute 'long_name' is missing or empty. The metadata standard sets units + long_name as the minimum. |
| ERROR | `attr.global.conventions.missing` | 12 | No Conventions attribute. The standard asks every file to name the convention it follows, with CF preferred. |
| WARN | `catalog.variable.undocumented` | 321 | 'dew_point' at WRF/1hr is not in variable_descriptions.csv at any resolution, so climakitae has no unit or display name for it. |
| WARN | `catalog.path.member_level_differs` | 80 | WRF paths from UCSD carry a member_id level that the rest of the activity omits. The layout is not consistent within the activity, so anything rebuilding paths  |
| WARN | `catalog.variable.timescale_mismatch` | 71 | 'rh' is published at WRF/1hr but variable_descriptions.csv lists it only at ['day', 'mon'], so no unit or display name resolves at this resolution. |
| WARN | `attr.global.empty` | 64 | 1 global attribute(s) are present but empty. A reader cannot tell an empty value from an unanswered question. |
| WARN | `dataset.aggregated` | 37 | This result combines several stores (sim=3) rather than being a single published dataset. Time alignment across members can drop dates that any one member lacks |
| WARN | `wind.rotation.undeclared` | 16 | 'uas' declares no reference frame in its attributes. WRF output is grid-relative by default, but a consumer has no way to confirm that from the file, and rotate |
| WARN | `attr.global.recommended.missing` | 15 | Missing 4 of 6 recommended global attributes; datasets are supposed to be self-describing. |
| WARN | `catalog.path.pseudo_institution` | 8 | Path uses 'derived-vars' in the institution slot while the catalog attributes this record to 'CAE'. A deliberate namespace for computed products, but it means p |
| WARN | `coverage.variable.absent` | 8 | 'p' is documented for WRF/1hr but no store exists at any grid label. |
| WARN | `coverage.variable.timescale_mismatch` | 4 | 'tasmax' is published at LOCA2/yrmax but the reference variable table lists it only at day,mon. climakitae will find no unit or display name for it here. |

## Datasets with the most findings

| activity_id | table_id | grid_label | variable_id | worst | n | codes |
| --- | --- | --- | --- | --- | --- | --- |
| WRF | 1hr | d02 | v10 | ERROR | 36 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| WRF | 1hr | d02 | u10 | ERROR | 36 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| WRF | 1hr | d01 | u10 | ERROR | 36 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| WRF | 1hr | d01 | v10 | ERROR | 36 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| WRF | 1hr | d03 | v10 | ERROR | 32 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| WRF | 1hr | d03 | u10 | ERROR | 32 | `catalog.path.staging_prefix, catalog.path.variable_qualifier_dropped` |
| LOCA2 | day | d03 | wspeed | ERROR | 31 | `attr.global.conventions.missing, attr.global.empty, attr.global.recommended.missing, crs.spatial_ref.missing, dataset.ag` |
| LOCA2 | day | d03 | uas | ERROR | 27 | `attr.global.conventions.missing, attr.global.empty, attr.global.recommended.missing, crs.spatial_ref.missing, dataset.ag` |
| LOCA2 | day | d03 | vas | ERROR | 27 | `attr.global.conventions.missing, attr.global.empty, attr.global.recommended.missing, crs.spatial_ref.missing, dataset.ag` |
| LOCA2 | day | d03 | hursmax | ERROR | 24 | `attr.global.empty, attr.var.long_name.missing, crs.spatial_ref.missing, dataset.aggregated` |
| WRF | 1hr | d01 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | 1hr | d02 | rh | WARN | 19 | `catalog.variable.timescale_mismatch` |
| WRF | 1hr | d02 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | 1hr | d01 | rh | WARN | 19 | `catalog.variable.timescale_mismatch` |
| WRF | day | d02 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | day | d01 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | mon | d01 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | mon | d02 | dew_point | WARN | 19 | `catalog.variable.undocumented` |
| WRF | 1hr | d02 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | 1hr | d01 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | day | d01 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | day | d02 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | mon | d01 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | mon | d02 | effective_temp_index | WARN | 18 | `catalog.variable.undocumented` |
| WRF | 1hr | d03 | rh | WARN | 17 | `catalog.variable.timescale_mismatch` |
