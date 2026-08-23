# Sea Fog Case Preview Images

This dataset contains preview images for sea fog cases over the Yellow Sea and Bohai Sea region. The images are intended to support manual inspection, case classification, and documentation of the sea fog events used in the associated study.

Each PNG file corresponds to one CALIPSO overpass time. File names preserve the original CALIPSO L1 time stamp, for example:

```text
CAL_LID_L1-Standard-V4-51.YYYY-MM-DDTHH-MM-SSZD_Subset_l1_modis_preview.png
```

## Directory Structure

```text
sea_fog_case_preview_images/
  calipso_validated_sea_fog/
    calipso_validation_cases/
    fragmented_sea_fog/
    insufficient_island_constraints/
    modis_swath_seam_affected_cases/
    synoptic_system_obscured_sea_fog/
  modis_identified_sea_fog/
```

## Categories

### calipso_validated_sea_fog

Sea fog cases verified using CALIPSO observations.

- `calipso_validation_cases`: Core CALIPSO validation cases.
- `fragmented_sea_fog`: CALIPSO-validated cases with spatially fragmented sea fog features.
- `insufficient_island_constraints`: CALIPSO-validated cases for which island-based fog-top height retrieval is limited by insufficient island constraints.
- `modis_swath_seam_affected_cases`: CALIPSO-validated cases affected by MODIS swath seams near the target fog area.
- `synoptic_system_obscured_sea_fog`: CALIPSO-validated sea fog cases strongly obscured by synoptic weather systems such as cyclones, fronts, or associated cloud systems.

### modis_identified_sea_fog

Sea fog or low-cloud/fog cases identified from MODIS imagery but not validated by CALIPSO along the fog area.

## File Counts

| Category | Count |
|---|---:|
| `calipso_validated_sea_fog/calipso_validation_cases` | 49 |
| `calipso_validated_sea_fog/fragmented_sea_fog` | 28 |
| `calipso_validated_sea_fog/insufficient_island_constraints` | 35 |
| `calipso_validated_sea_fog/modis_swath_seam_affected_cases` | 8 |
| `calipso_validated_sea_fog/synoptic_system_obscured_sea_fog` | 58 |
| `modis_identified_sea_fog` | 501 |
| **Total** | **679** |

## Notes

- The preview images are for visual documentation and manual quality control.
- CALIPSO-validated cases are separated from MODIS-identified cases to distinguish lidar-confirmed sea fog from cases inferred primarily from passive satellite imagery.
- The category names reflect the final manual classification used for the supporting dataset.


## JPEG Version

This folder contains JPEG-converted versions of the original PNG preview images. The category structure and file-name stems are preserved.
