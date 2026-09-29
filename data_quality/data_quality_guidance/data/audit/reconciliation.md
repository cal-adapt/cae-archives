# cadcat: documentation compliance across both surfaces

Compared **40** dataset(s) across two surfaces: the Zarr stores published in `s3://cadcat`, and the objects `climakitae.ClimateData` returns to users.

Both are judged against the Cal-Adapt documentation. A requirement can fail on one surface and pass on the other, which is why each row records where it failed rather than a single verdict.

| verdict | rows |
| --- | ---: |
| introduced_by_reader | 46 |
| both | 60 |
| store_defect_repaired | 40 |
| compliant | 200 |

200 check(s) passed identically on both surfaces and are listed in the CSV with verdict `compliant`; the sections below cover only divergences and shared failures.

## Introduced by climakitae

The published store satisfies the requirement; what users receive does not. Every Analytics Engine user is affected regardless of the store's state, and fixing the store would change nothing.

| code | n | documented requirement | source |
| --- | ---: | --- | --- |
| `dataset.aggregated` | 37 | A query pinning every available facet should identify one store. | Data Structure and Format |
| `attr.global.recommended.missing` | 3 | Recommended global attributes: title, institution, source, references, history, comment. | Metadata Standards |
| `time.gaps` | 3 | The time variable must represent time elapsed since a reference date; Gregorian is preferred. | Metadata Standards |
| `time.leap.calendar_contradiction` | 3 | Documented leap-day behaviour per model; LOCA2 models were interpolated to include leap days. | Climate Model Simulations |

## Non-compliant on both surfaces

Neither the store nor the delivered product meets the documented requirement. Fix at the source.

| code | n | documented requirement | source |
| --- | ---: | --- | --- |
| `attr.global.empty` | 32 | Datasets should be self-describing; an attribute present but empty describes nothing. | Metadata Standards |
| `attr.var.long_name.missing` | 8 | Variable attributes must contain a long name descriptor, descriptive enough to label plots. | Metadata Standards |
| `wind.rotation.undeclared` | 8 | Wind components must declare their reference frame; grid-relative and earth-relative are not interchangeable. | Climate Model Simulations |
| `attr.global.conventions.missing` | 6 | Each file must list the standard convention used to organize the data; CF Convention is preferred. | Metadata Standards |
| `attr.global.recommended.missing` | 6 | Recommended global attributes: title, institution, source, references, history, comment. | Metadata Standards |

## Repaired by climakitae

The store does not meet the requirement, but climakitae compensates on read, so users do not see it. Still a defect in the published artifact: it affects anyone reading the Zarr directly, and it depends on the repair continuing to exist.

| code | n | documented requirement | source |
| --- | ---: | --- | --- |
| `crs.spatial_ref.missing` | 40 | WRF carries a grid_mapping attribute referencing a Lambert_Conformal coordinate variable; LOCA2-Hybrid stores CRS in a spatial_ref coordinate at the global level. | Climate Model Simulations |

## Documentation

- [Metadata Standards](https://analytics.cal-adapt.org/data-tools/data-documentation/metadata-standards.html)
- [Data Structure and Format](https://analytics.cal-adapt.org/data-tools/data-documentation/data-structure-and-format.html)
- [Climate Model Simulations](https://analytics.cal-adapt.org/data-tools/data-documentation/climate-model-sims.html)
- [climakitae variable_descriptions.csv](https://github.com/cal-adapt/climakitae)
