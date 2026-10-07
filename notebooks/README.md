# notebooks/ — analysis and structure-construction notebooks

All notebooks use paths relative to this folder (`../data/...`). Run them with the kernel of the
environment in `../requirements.txt`. Plots that are not final paper figures are written under
`../data/figures/` (git-ignored, regenerable).

## Analysis of the DFT results (run order: top to bottom)

| notebook | reads | writes | purpose |
|---|---|---|---|
| `dft_convexhull_plot.ipynb` | `singleTM_tighter_conv.csv` (+ `singleTM_w_exp_2nd_relax.csv` for comparison) | `data/figures/singleTM/convexhull_*.pdf` | formation energy per TMO2 vs x and lower convex hull, single TM. Contains the `na_map` that converts ordering labels to x. |
| `dft_convexhull_doubleTM.ipynb` | `doubleTM_dft_tighter.csv` | `data/figures/doubleTM/convexhull_*.pdf` | same for the ten mixed systems (both `elec` configurations on one hull) |
| `dft_structure_and_thermo_analysis.ipynb` | the two tables above, `*_structures.json`, `experiments/*.csv` | `singleTM_hull_points.csv`, `doubleTM_hull_points.csv`, `*_hull_mp_ehull.csv`, XRD PDFs | **step 1** extracts the hull-stable points (authoritative ground-state list); **step 2** places our PBE+U energies on the Materials Project hull (MP2020 compatibility, `MPRester`; needs `export MP_API_KEY=...`); **step 3** simulates XRD of every hull structure. |
| `dft_voltagecurve_plot.ipynb` | `singleTM_tighter_conv.csv`, `singleTM_magmom.csv`, `experiments/*.csv` | `data/figures/singleTM/voltage_*.pdf` | 0 K voltage profiles vs Na/Na+ from the hull, overlaid with the digitized experimental curves; optional magnetic-moment panel |
| `dft_voltagecurve_doubleTM.ipynb` | `doubleTM_dft_tighter.csv`, `doubleTM_magmom.csv`, `experiments/*.csv` | `data/figures/doubleTM/voltage_*.pdf` | same for the mixed systems |
| `dft_oxidation_analysis.ipynb` | `doubleTM_dft_tighter.csv`, `doubleTM_magmom.csv` | `doubleTM_stable_oxi_states.csv`, moment histograms | assigns oxidation states site by site from the on-site magnetic moments (moment windows per metal / spin state, ambiguous cases resolved by cell charge neutrality) and averages them per element; the basis of Figure 3, Figure 4 and Table S1 |

## Construction of the ordering library

| notebook | purpose |
|---|---|
| `O3_ordering_generation.ipynb` | cuts a Na/vacancy ordering cell out of the 7×7×1 O3-NaCoO2 host (`data/O3_orderings/O3_NaCoO2_POSCAR_771.vasp`) from four chosen Na sites; this is how the `AJT_*` and `JLK_*` O3 orderings were built from the published diagrams |
| `P3_ordering_generation.ipynb` | P3 analogue (host `data/P3_orderings/P3_NaCoO2_771.vasp`; keeps z instead of shifting) |
| `misc.ipynb` | last section: converts the experimental CIFs in `data/experiments/orderings/` (1×1×3 supercell) into the `JMAS_*`, `XC_*`, `CD_a` POSCARs. The earlier cells are MACE sanity checks from the parent project and are not needed here. |
| `Staging_layers_noTMMix.ipynb` | enumerates the three-layer stackings of each `*_w_vac` ordering: finds every in-plane translation that maps the template Na layer onto the other two layers, builds all combinations, removes symmetry-equivalent ones with `StructureMatcher`, and writes `data/mace_screening/singleTM/ordering_staging/<phase>/<ordering>/group_NNN/`. The groups are then relaxed with MACE by `scripts/mace_ordering_calculation_cli.py`. |

## Changes relative to the working copies

* `dft_structure_and_thermo_analysis.ipynb`: the Materials Project API key is read from the
  environment (`MP_API_KEY` or `PMG_MAPI_KEY`) instead of being written in the cell.
* `dft_voltagecurve_plot.ipynb`: the cells comparing DFT with MACE trajectories (parent project)
  and two scratch cells were removed; the bold `$\bf{...}$` legend labels, which had lost their
  backslash, were restored so the styled-plot cell runs again.
* `Staging_layers_noTMMix.ipynb`, `dft_oxidation_analysis.ipynb`: absolute paths replaced by relative ones.

Five of the six `dft_*` notebooks were executed end to end from this folder on 2026-10-07.
`dft_structure_and_thermo_analysis.ipynb` was not re-executed because its step 2 needs a Materials
Project API key; its stored outputs are the ones that produced the `data/*_hull_*.csv` files.
