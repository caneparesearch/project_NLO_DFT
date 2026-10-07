# project_NLO_DFT — DFT ground-state map of O3/P3/O1 Na layered oxides

Code, DFT datasets, and Na/vacancy ordering library behind the manuscript

> *Ordering-resolved DFT map of single- and mixed-transition-metal Na layered oxides
> (NaxTMO2 and NaxTM0.5TM'0.5O2, TM = Ti, Mn, Fe, Co, Ni) in the O3 / P3 / O1 stacking family.*

Everything needed to **regenerate every figure and number in the paper from the tabulated DFT
results** is here. Raw VASP outputs (≈140 GB) are not included; the relaxed structures, energies,
magnetic moments, and derived quantities are tabulated in `data/`.

This repository is the "essential" subset split out of a larger working repository
(`project_NaLayered_MLIP`) that also contains MACE fine-tuning, phonon, and Na-diffusion work.
Nothing from that MLIP side is included here.

---

## 1. Layout

```
project_NLO_DFT/
├── README.md                ← this file (overview, workflow, how to reproduce)
├── requirements.txt         ← Python packages (versions used for the final figures)
├── data/                    ← all inputs and tabulated results      → data/README.md
│   ├── O1_orderings/, O3_orderings/, P3_orderings/   Na/vacancy ordering library (POSCAR, 41 structures)
│   ├── experiments/         digitized experimental voltage curves (CSV) + refined CIFs
│   ├── mace_screening/      stacking candidates + MACE pre-screening results that fed DFT
│   ├── singleTM_*.csv/json  DFT results, 5+2 single-TM systems (287 runs)
│   └── doubleTM_*.csv/json  DFT results, 10 mixed-TM systems (820 runs)
├── figures/                 ← figure scripts and the final PDFs       → figures/README.md
│   ├── codes/               one script per figure (Figure2.py, Figure3.py, …)
│   ├── *.pdf                final figure files as used in the manuscript
│   └── convex_hulls/, xrd_patterns/, van_vleck/, …   per-system panels and assembly .pptx
├── notebooks/               ← Jupyter notebooks: hull / voltage / oxidation / XRD / MP-ehull analysis,
│                               ordering construction, stacking enumeration      → notebooks/README.md
└── scripts/                 ← workflow scripts used on the cluster (ordering enumeration, MACE
                                pre-screening, VASP input generation, result collection) → scripts/README.md
```

Each sub-directory has its own README describing every file in it.

---

## 2. Quick start — regenerate the figures

```bash
# 1) environment (Python 3.10)
conda create -n nlo python=3.10
conda activate nlo
pip install -r requirements.txt

# 2) VanVleckCalculator (only for the Van Vleck figure; not on PyPI)
git clone https://github.com/lnaglecocco/VanVleckCalculator.git ../VanVleckCalculator
#    (or: export VVC_CODE=/path/to/VanVleckCalculator/code)

# 3) regenerate any figure — scripts locate data/ relative to their own path
cd figures/codes
python Figure2.py          # → ../Figure2_voltage_caxis.pdf
python FigureS3S4.py       # → ../convex_hulls/convexhull_*.pdf
```

The full figure → script map, and the required run order for the Van Vleck chain, is in
[`figures/README.md`](figures/README.md). All scripts run in a few seconds; the slowest
(`FigureS5S6.py`, simulated XRD of every hull structure) takes about 20 s.

Only `notebooks/dft_structure_and_thermo_analysis.ipynb` (energy above the Materials Project hull)
needs network access; set `MP_API_KEY` in your environment before running that step.

---

## 3. Workflow overview

```
 [A] Ordering library                      data/O3_orderings, P3_orderings, O1_orderings
     41 Na/vacancy orderings from the literature (Toumar 2015 = AJT, Kaufman & Van der Ven 2019/2022 = JLK,
     Steele 2025/2026 = JMAS, Chen 2018 = XC, Didier 2012 = CD, Li 2014 = XL) + full/empty hosts.
     Built with notebooks/O3_ordering_generation.ipynb, P3_ordering_generation.ipynb (cell cutting from
     7×7 host supercells) and notebooks/misc.ipynb (experimental CIF → POSCAR).
          │
 [B] Layer-stacking enumeration            data/mace_screening/singleTM/ordering_staging
     Each ordering is placed in a 3-Na-layer cell; the reference layer is propagated to the other two
     layers by every in-plane translation; symmetry duplicates removed.
     notebooks/Staging_layers_noTMMix.ipynb
          │
 [C] MACE-OMAT-0 pre-screening             data/mace_screening/singleTM/ordering_staging_output
     Every stacking × TM (Ti, V, Cr, Mn, Fe, Co, Ni) is relaxed with MACE-OMAT-0 (medium); the lowest
     stacking per (ordering, TM) goes to DFT.   scripts/mace_ordering_calculation_cli.py
          │
 [B'/C'] Mixed-TM systems                  data/mace_screening/doubleTM/ordering_doubleTM
     1:1 in-plane TM mixing ranked by Ewald energy (top electrostatic configurations elec_0, elec_1),
     TM-layer stacking then Na-layer stacking selected by single-point MACE energies.
     scripts/util_tmmix_tmstaging.py → mace_eval_tmmix_tmstaging.py → util_tmmix_nastaging.py → mace_eval_tmmix_nastaging.py
          │
 [D] DFT (VASP, PBE+U, MPRelaxSet-based)   scripts/dft_prep_*.py, dft_generate_*.py
     ENCUT 680 eV, KSPACING 0.3 Å⁻¹, EDIFF 1e-6 eV/atom, EDIFFG −0.02 eV/Å, DFT-D3(BJ), ferromagnetic start.
     Hull-stable structures re-relaxed with default MPRelaxSet (520 eV) for the Materials-Project ehull.
          │
 [E] Collection                            scripts/dft_collect_*.py  →  data/*.csv, data/*_structures.json
          │
 [F] Analysis                              notebooks/dft_*.ipynb  →  data/*_hull_points.csv, *_hull_mp_ehull.csv,
     convex hulls, voltage profiles,                                   doubleTM_stable_oxi_states.csv
     oxidation states from magnetic moments, XRD, MP ehull
          │
 [G] Figures                               figures/codes/*.py  →  figures/*.pdf
```

---

## 4. Systems covered

| set | systems | DFT runs (`*_tighter*.csv`) | hull points |
|---|---|---|---|
| single TM | Ti, Mn, Fe, Co, Ni (+ Cr, V computed but not in the paper) | 287 | 48 |
| mixed TM 1:1 | Ti-Mn, Ti-Fe, Ti-Co, Ti-Ni, Mn-Fe, Mn-Co, Mn-Ni, Fe-Co, Fe-Ni, Co-Ni | 820 | 69 |

Keys used throughout the data files:

* `phase` ∈ {`O1_orderings`, `O3_orderings`, `P3_orderings`} — host stacking.
* `ordering` — name of the Na/vacancy ordering, e.g. `AJT_b_w_vac`, `JLK_zeta`, `JMAS_c`, `O3_full`, `empty`.
* `tm` (single) or `tm_pair` + `elec` (mixed; `elec_0`/`elec_1` = Ewald-ranked in-plane TM configuration).
* `xNa` — Na content per TMO2 formula unit.
* structure key = `phase-ordering-tm` or `phase-ordering-tm_pair-elec` (index into `*_structures.json`).

---

## 5. What is deliberately **not** here

* Raw VASP directories (INCAR/OUTCAR/vasprun.xml/CONTCAR). The settings are fully specified by
  `scripts/dft_prep_from_mace_*.py` and the relaxed structures are in `data/*_structures.json`.
* The MACE fine-tuning datasets and models, phonon calculations, and slab-gliding work of the parent repository.
* Source images (PNG/PDF) of the experimental papers from which `data/experiments/*.csv` were digitized
  (copyright); only the digitized numbers are kept.
* Manuscript files.

## 6. Notes on provenance / known quirks

* The figure scripts keep an older internal numbering: `Figure4a–e.py` and `Figure4_setting.py` produce
  **paper Figure 7** (stacking descriptors) and SI Figure S5; `Figure5*.py` produce **paper Figure 6**
  (energy-density vs strain Pareto map); `Figure4_vanvleck*.py` produce **paper Figure 5**;
  `FigureS_redox_*.py` produce **paper Figure 4**. See `figures/README.md`.
* Some orderings are duplicates imported from different papers: XC_a = JMAS_a = AJT_d (x = 1/3),
  JMAS_c = JLK_zeta (x = 1/2), AJT_g = CD_a (x = 1/2), JMAS_b = AJT_f (x = 2/5). AJT_j (x = 5/8) is
  excluded (Na off the ideal sites, no structural data).
* The `*_w_vac.vasp` ordering files mark vacant Na sites with **K**; the workflow removes K before any
  calculation. Files without `_w_vac` are the same supercell with every Na site filled.
* Three DFT datasets exist for the mixed systems: `doubleTM_dft_summary.csv` (first relaxation),
  `doubleTM_dft_2nd_relax_summary.csv` (second MPRelaxSet relaxation, used for the MP ehull), and
  `doubleTM_dft_tighter.csv` (tight-convergence set, **the hull/figure dataset**). Use the `tighter`
  files unless you specifically want the others.
