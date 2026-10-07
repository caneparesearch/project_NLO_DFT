# scripts/ — workflow scripts (structure enumeration → MACE pre-screening → VASP → collection)

These are the command-line scripts that were run on the cluster to produce `data/`. They are kept
so the pipeline is documented and re-runnable; **none of them is needed to regenerate the figures**.
Each script has a module docstring with the expected directory tree and full usage; `python <script> -h`
lists the options. Default paths assume the layout of the working repository
(`../data`, `../ordering_staging`, `../dft_singleTM`, …); pass explicit `--root/--out-root` arguments here.

## Pipeline

```
single TM                                                   mixed TM (1:1)
─────────                                                   ──────────────
notebooks/Staging_layers_noTMMix.ipynb                      util_tmmix_tmstaging.py      (step1: Ewald-ranked in-plane TM mixing
  → ordering_staging/<phase>/<ordering>/group_*/POSCAR         of one layer; step2: propagate to the other
                                                               TM layers by translation; step3: group by symmetry)
mace_ordering_calculation_cli.py                            mace_eval_tmmix_tmstaging.py (single-point MACE → POSCAR_TMstaging)
  (strip K, Co→TM, MACE relax, summary.csv)                 util_tmmix_nastaging.py      (enumerate Na-layer translations)
util_full_na_phase_pipeline.py (full/empty hosts)           mace_eval_tmmix_nastaging.py (single-point MACE → POSCAR_lowestEconfig.vasp)
        │                                                           │
dft_prep_from_mace_singleTM.py                              dft_prep_from_mace_doubleTM.py
dft_prep_from_exp_singleTM.py (experimental orderings)              │
        └──────────────────────── VASP relax ───────────────────────┘
                 dft_generate_2ndrelaxation_run.py  (default MPRelaxSet re-relax → MP ehull)
                 dft_generate_static_run.py         (tight single point)
                                   │
   dft_collect_from_{single,double}TM.py        → data/*_dft_*.csv   (energies, convergence, custodian diagnostics)
   dft_collect_CONTCAR_from_{single,double}TM.py → data/*_structures.json
   dft_collect_magmom_{single,double}TM.py       → data/*_magmom.csv
```

## Scripts

### Ordering / stacking enumeration
| script | what it does |
|---|---|
| `util_tmmix_tmstaging.py` | Mixed-TM ordering generator. For one TM pair and every ordering POSCAR: mixes the first TM layer 1:1 (cell doubled in plane if the layer has an odd site count), ranks the in-plane configurations by Ewald energy (`--max-ranked`, keeps `--top-elec`), propagates the pattern to the other TM layers by lattice translations, and groups the stackings with `StructureMatcher`. SLURM-shardable (`SLURM_PROCID`/`SLURM_NTASKS`) with multiprocessing. Output `ordering_doubleTM/<phase>/<ordering>/<TM_pair>/elec_*/{POSCAR_step1, TM_staging/group_*/POSCAR, step1_meta.json}` and `target_layer_data.json`. |
| `util_tmmix_nastaging.py` | After the TM stacking is chosen: enumerates the Na-layer translations of the K-marked template layer for every `elec_*/POSCAR_TMstaging` → `Na_staging/trans_*/POSCAR`. |
| `util_map_expordering.py` | Helper used when importing experimental orderings: groups the Na sites of a structure by their six nearest-O fractional vectors to check site equivalence (`<stem>_na_o_vectors.json`). |
| `util_full_na_phase_pipeline.py` | Endpoint sweep without Na ordering: full/empty O3, empty O1, Co → each TM, MACE relax, appends to `summary.csv`, writes VASP inputs. |

### MACE-OMAT-0 pre-screening (needs `mace-torch` and the `mace-omat-0-medium.model` checkpoint)
| script | what it does |
|---|---|
| `mace_ordering_calculation_cli.py` | For every `group_*` POSCAR under `--staging-root`: remove K, substitute Co by each TM in {Ti, V, Cr, Mn, Fe, Co, Ni}, relax ions (+cell with `--relax-cell`) with MACE; writes `final.POSCAR`, per-case `summary.json`, global `summary.csv` (→ `data/mace_screening/singleTM/ordering_staging_output`). |
| `mace_eval_tmmix_tmstaging.py` | Single-point MACE energies of all `TM_staging/group_*` candidates per `elec_*`, picks the lowest → `POSCAR_TMstaging` + `TM_staging_mace_summary.json` (K temporarily swapped to Na for the evaluation). |
| `mace_eval_tmmix_nastaging.py` | Same for `Na_staging/trans_*` → `POSCAR_lowestEconfig.vasp` + `Na_staging_mace_summary.json`. |

### VASP input generation (pymatgen `MPRelaxSet`, PBE_54 POTCARs)
| script | settings |
|---|---|
| `dft_prep_from_mace_singleTM.py` | picks the lowest-energy MACE group per (phase, ordering, TM) from `summary.csv`, writes `<out_root>/<phase>/<ordering>/<tm>/{POSCAR, INCAR, POTCAR, job.sh, MACE_metadata.txt}` and `minima_summary.csv`. INCAR overrides: ENCUT 680, EDIFF 1e-6 eV/atom, EDIFFG −0.02, KSPACING 0.3 (no KPOINTS file), ISPIN 2 ferromagnetic, IVDW 12 (D3-BJ); Hubbard U from MPRelaxSet. |
| `dft_prep_from_mace_doubleTM.py` | same inputs from `elec_*/POSCAR_lowestEconfig.vasp` → `<out_root>/<phase>/<ordering>/<TM1_TM2>/elec_*/`. |
| `dft_prep_from_exp_singleTM.py` | same inputs for the experimental orderings (all non-Na/O sites → each TM); records `POSCAR_exp_source` and `EXP_metadata.txt`. |
| `dft_generate_2ndrelaxation_run.py` | `2nd_relax/` under each converged run: CONTCAR → POSCAR, default MPRelaxSet (ENCUT 520, IVDW 12, ISPIN 2). This is the "MP-compatible" relaxation used for the energy above the Materials Project hull. |
| `dft_generate_static_run.py` | `static/` single point from a relax or 2nd_relax: ENCUT 520, dense Γ-centred mesh (kppa 6000), ISMEAR −5, EDIFF 1e-7/atom, NSW 0. |

### Result collection
| script | output |
|---|---|
| `dft_collect_from_singleTM.py` / `dft_collect_from_doubleTM.py` | one row per run with energies, lattice, convergence flags, custodian handler diagnostics (optionally `--apply-fixes` / `--submit-fixed` to resubmit); `--run-type relax|2nd_relax|static|r2scan` selects the sub-run. → `data/singleTM_tighter_conv.csv`, `doubleTM_dft_tighter.csv`, `*_dft_summary.csv`, `*_2nd_relax_summary.csv`. |
| `dft_collect_CONTCAR_from_singleTM.py` / `_doubleTM.py` | JSON of `Structure.as_dict()` keyed `phase-ordering-tm[_pair-elec]` → `data/*_structures.json`. |
| `dft_collect_magmom_singleTM.py` / `_doubleTM.py` | per-TM-site magnetic moments from OUTCAR → `data/*_magmom.csv`. |

### `slurm/`
Batch scripts as submitted (GPU partition for MACE, CPU nodes for the Ewald enumeration):
`run_batch_ordering_cli.slurm`, `run_batch_endpoints_cli.slurm`, `run_mace_eval_tmmix_{tm,na}staging.slurm`,
`run_tmmix_tmstaging.slurm` and its ten per-pair copies in `TMmix_scripts/`. Module names and
partitions are specific to the cluster used and will need editing.
