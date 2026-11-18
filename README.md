# Chemistry by Design (CbyD): analysis and route dashboard

Code for the CbyD paper (in preparation). There are two parts:

- `analysis/` makes the figures and tables for the paper.
- `dashboard/` builds the route viewer. The built page is published in the [CbyD data repository](https://github.com/chemistry-by-design/cbyd).

## Data

The CbyD routes are in the `data/routes/` folder of the [data repository](https://github.com/chemistry-by-design/cbyd). From the root of `cbyd_analysis`, copy them into `data/routes/`:

```bash
git clone https://github.com/chemistry-by-design/cbyd.git ../cbyd
mkdir -p data && cp -r ../cbyd/data/routes data/routes
```

The dashboard only needs `data/routes/`. The other datasets are used for comparison in the analysis. We don't redistribute them, because of licensing (some are commercial), so please download each one from its original source. Put them at the paths below, relative to `data/` (set in `analysis/utils/paths.py`):

| Data       | Source                                                          | File                                    |
| ---------- | --------------------------------------------------------------- | --------------------------------------- |
| CbyD       | [data repo][cbyd]                                               | `routes/`                               |
| PaRoutes   | [Genheden & Bjerrum 2022][paroutes]                             | `other_data/paroutes_n1.json`           |
| USPTO-190  | [Chen *et al.* 2020][retro]                                     | `other_data/uspto190.pkl`               |
| AbSynth    | [Czuba *et al.* 2026][absynth]                                  | `other_data/ABSynth_Dataset.csv`        |
| USPTO      | [Lowe 2017][lowe]; [Tran *et al.* 2026][tran]; [Zenodo][zenodo] | `other_data/uspto_clean.json`           |
| Pistachio  | [NextMove][pistachio]; [Tran *et al.* 2026][tran]               | `other_data/pistachio2025Q2_clean.json` |
| eMolecules | [download 2026-09-01][emol]                                     | `other_data/emolecules_buyable.smi`     |

[cbyd]: https://github.com/chemistry-by-design/cbyd
[paroutes]: https://doi.org/10.1039/d2dd00015f
[retro]: https://doi.org/10.48550/arXiv.2006.15820
[absynth]: https://doi.org/10.1021/jacs.6c08434
[lowe]: https://doi.org/10.6084/m9.figshare.5104873.v1
[tran]: https://doi.org/10.1039/d6sc01323f
[zenodo]: https://zenodo.org/records/18623634
[pistachio]: https://www.nextmovesoftware.com/pistachio.html
[emol]: https://downloads.emolecules.com/free/2026-09-01/

## Environment

```bash
conda env create -f environment.yml
conda activate cbyd_analysis
```

## Analysis

### 1. Precompute tables (run once)

These scripts extract reactions, molecules and route statistics from each dataset and save them as tables in `data/data_analysis/` for the notebook to read.

```bash
conda activate cbyd_analysis

# CbyD tables, route topology and route details
python analysis/scripts/00_prepare_tables.py --only cbyd
python analysis/scripts/00_prepare_tables.py --only targets
python analysis/scripts/30_route_topology.py
python analysis/scripts/30_route_topology.py --only absynth   # AbSynth routes
python analysis/scripts/50_route_details.py
python analysis/scripts/00_prepare_tables.py --only absynth   # AbSynth reactions and products
python analysis/scripts/50_route_details.py --dataset absynth # AbSynth starting materials and depth

# CbyD reaction features, NameRxn classes and molecular complexity
python analysis/scripts/20_rxn_features.py \
    --input data/data_analysis/reactions/cbyd_reactions.jsonl \
    --output data/data_analysis/reaction_features/cbyd_rxn_features.csv
python analysis/scripts/40_namerxn.py \
    --input data/data_analysis/reactions/cbyd_reactions.jsonl \
    --output data/data_analysis/reaction_class/cbyd_namerxn.csv --source cbyd
for f in cbyd_products cbyd_targets uspto190_targets paroutes_targets absynth_targets absynth_products; do
  python analysis/scripts/10_mol_complexity.py \
    --input data/data_analysis/molecules/$f.jsonl \
    --output data/data_analysis/complexity/${f}_complexity.csv
done

# Single-step comparison sets
bash analysis/scripts/run_full_onestep.sh 48
```

### 2. Run the notebook

Open `analysis/notebook/comparative_analysis.ipynb` with the `cbyd_analysis` kernel and run all cells. 

## Dashboard

Builds `dashboard.html`, a single page for browsing the CbyD routes that opens in a browser without a server. Each route is drawn as a tree, and clicking a reaction shows its mapped SMILES and reagents.

```bash
conda activate cbyd_analysis
python dashboard/build_static.py
```

## Citation

To be added on publication.

```bibtex
@article{xxx,
  title   = {xxx},
  author  = {xxx},
  journal = {xxx},
  year    = {xxx},
  doi     = {xxx}
}
```
