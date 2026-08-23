# Sea Fog Top Height Retrieval Using Island Elevation

This repository implements a sea fog top height retrieval workflow for the
Yellow and Bohai Seas. It combines MODIS Band 3 reflectance, island peak
elevation from NASADEM, island visibility categories, and CALIPSO-derived fog
top height observations.

The current validation set contains 49 sea fog cases. MODIS cloud-top height
is used as a benchmark, while logistic regression (LR) and a linear support
vector machine (SVM) derive reflectance-height decision boundaries from visible
and obscured islands.

## Workflow

For each case, the program:

1. Loads MODIS Band 3 reflectance, MODIS cloud-top height, a MODIS preview
   image, CALIPSO fog-top height, and island visibility labels.
2. Extracts representative reflectance around each selected island.
3. Fits LR and linear-SVM classification boundaries in reflectance-elevation
   space.
4. Converts each boundary into a pixel-wise fog-top height estimate.
5. Samples MODIS, LR, and SVM estimates along the CALIPSO track.
6. Calculates MAE, MAPE, RMSE, and hit rate for each case and for all cases.
7. Generates individual and combined publication figures.

The hit rate is the percentage of estimates with an absolute error no greater
than 30 m.

## Validation Data

The active case list is:

```text
filelist_49cases.csv
```

It contains one relative path to a MODIS Band 3 NPY file per row. The list has
49 entries and no header.

Island peak elevations are read from:

```text
island_info_nasadem.csv
```

Expected columns are:

```csv
id,lon,lat,pe,area
```

Here, `pe` is the NASADEM-derived island peak elevation in metres and `area`
is used to determine the reflectance sampling window.

Each case also requires an island category file:

```text
islands_categories/YYYYDDD.csv
```

Expected columns include:

```csv
id,pe,category
```

Category 1 denotes a visible island and category 2 denotes an island obscured
by fog or low cloud. The retrieval uses `id` and `category`; elevations are
taken from `island_info_nasadem.csv` to ensure a consistent source.

## Required Directory Structure

```text
.
|-- main.py
|-- config.py
|-- data_loader.py
|-- models.py
|-- visualization.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- filelist_49cases.csv
|-- island_info_nasadem.csv
|-- islands_categories/
|-- modis_b03/
|-- modis_cths/
|-- modis_imgs/
|-- calipso_sfths/
|-- scatter_figs/
|-- height_compare/
|-- height_image/
`-- calipso_footprints/
```

The program creates the four output directories automatically if they do not
already exist.

### Input File Conventions

- `modis_b03/`: MODIS Band 3 reflectance arrays in NPY format. Both legacy
  `*_B03.npy` and current `*.B03.npy` names are supported.
- `modis_cths/`: MODIS cloud-top height arrays in NPY format. Files are matched
  by the `YYYYDDD` identifier.
- `modis_imgs/`: MODIS preview images. PNG, JPG, and JPEG are supported and are
  matched to the Band 3 filename.
- `calipso_sfths/`: CSV files named `YYYYDDD_calipso_result.csv`. The three
  columns are latitude, longitude, and fog-top height in metres.
- `islands_categories/`: One CSV file per case, named `YYYYDDD.csv`.

Missing MODIS cloud-top heights are converted from NaN to zero. Non-finite
Band 3 values are excluded when island reflectance is sampled. A case is
skipped with an error message if a required file or valid island sampling
window is unavailable.

## Requirements

The tested environment uses Python 3.12.3. Install the pinned direct
dependencies with:

```bash
python -m pip install -r requirements.txt
```

The direct third-party dependencies are:

```text
numpy
pandas
scikit-learn
scikit-image
matplotlib
seaborn
cartopy
shapely
pillow
```

For environments where Cartopy is difficult to install with pip, a conda-forge
environment is recommended:

```bash
conda install -c conda-forge numpy pandas scikit-learn scikit-image matplotlib seaborn cartopy shapely pillow
```

The figures use Times New Roman when the font is available.

## Configuration

Spatial settings and data directories are defined in `config.py`:

```python
max_lat = 42
min_lat = 30
max_lon = 129
min_lon = 117

res_modis = 0.05
res_radiance = 0.005
```

The active sample list and NASADEM island file are selected in `main.py`:

```python
filename = 'filelist_49cases.csv'
islands_info = pd.read_csv('island_info_nasadem.csv', sep=',')
```

Plot generation is controlled by the local `is_plot` variable in `main()`.
Set it to `False` when only numerical results are needed.

## Running the Retrieval

Run the workflow from the repository root:

```bash
python main.py
```

The console reports per-case and overall metrics for MODIS, LR, and SVM. A
failed case is reported and skipped without stopping the remaining cases.

## Outputs

### Root-Level Summary Figures

- `3d_island_height.pdf`: NASADEM island peak elevation distribution.
- `bias_violin.pdf`: overall error distributions for MODIS, LR, and SVM.
- `bias_violin_by_date.pdf`: 5 by 10 panel plot containing Total plus 49 cases.
- `method_all_metrics_heatmap_grouped.png`: MAE, RMSE, and hit rate for 49 cases
  plus the overall average.

### Per-Case and Combined Figures

- `scatter_figs/YYYYDDD.png`: island reflectance-elevation scatter plot.
- `scatter_figs/all_cases_scatter_combined.pdf`: 7 by 7 scatter layout for the
  49 cases.
- `calipso_footprints/YYYYDDD.pdf`: MODIS image, CALIPSO track, fog points, and
  island states.
- `calipso_footprints/all_cases_tracks_combined.pdf`: 7 by 7 track layout for
  the 49 cases.
- `height_compare/YYYYDDD.pdf`: CALIPSO, MODIS, LR, and SVM height comparison
  along the track.
- `height_image/YYYYDDD_LR.pdf`: LR fog-top height map.
- `height_image/YYYYDDD_SVM.pdf`: SVM fog-top height map.
- `height_image/YYYYDDD_MODIS.pdf`: MODIS cloud-top height map.

## Modules

- `main.py`: case processing, evaluation, and workflow orchestration.
- `config.py`: paths, study area, resolutions, method names, and colours.
- `data_loader.py`: MODIS, CALIPSO, category, and island-reflectance loading.
- `models.py`: LR and linear-SVM decision-boundary models.
- `utils.py`: metrics, date conversion, and combined-track data collection.
- `visualization.py`: individual and combined publication figures.

## Notes on Reproducibility

- The code does not download MODIS, CALIPSO, or NASADEM data automatically.
- Satellite data redistribution may be subject to the policies of the original
  data providers.
- `excluded_cases/`, if present locally, is an archive of cases not referenced
  by `filelist_49cases.csv`; it is not read by `main.py`.
- The current workflow assumes the grid geometry and resolutions specified in
  `config.py`. Verify these values before using a different gridded product.

## Citation

If this code is used in a publication, cite the associated manuscript and the
original MODIS, CALIPSO, and NASADEM data products. Add the final manuscript
citation here when it becomes available.

## License

No license file is currently included. Add an appropriate `LICENSE` file before
redistributing or accepting external contributions.
