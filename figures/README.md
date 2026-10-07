# figures/ — figure scripts and final figure files

`codes/` holds one Python script per figure (or per figure family). Every script locates the data
with `DATA_DIR = <repo>/data` relative to its own location and writes its PDF next to this README
(or into the sub-folder named below), so it can be run from anywhere:

```bash
cd figures/codes && python Figure2.py
```

The scripts share a common plot style (`rcParams` block at the top of `Figure2.py`: Arial-like
sans fonts, inward ticks, PDF fonttype 42 so the text stays editable in Illustrator).

> **Numbering.** The script names keep the figure numbers of an earlier draft. The table below maps
> them to the **final manuscript** numbering.

## Main-text figures

| paper | content | script(s) | output |
|---|---|---|---|
| Figure 1 | O1 / O3 / P3 stacking models and Na site networks | drawn by hand: VESTA renders `ordering_O1O3P3/{O1,O3,P3}_ordering.png`, assembled in `ordering_O1O3P3/collect.pptx` | `Figure1_full_labeled.pdf` |
| Figure 2 | voltage profiles + volume + c-axis per layer, 5×5 lower-triangular matrix, with experimental curves | `Figure2.py` | `Figure2_voltage_caxis.pdf` |
| Figure 3 | average TM oxidation states vs x, coloured by energy above the MP hull | `Figure3.py` | `Figure3_oxidation_table.pdf` |
| Figure 4 | (a) redox-couple voltage ladder, (b) schematic band diagram | `FigureS_redox_ladder.py` (a), `FigureS_redox_band_schematic.py` (b); assembled in `oxidation_states/collect.pptx` | `FigureS_redox_ladder.pdf`, `FigureS_redox_band_schematic.pdf` → `Figure4_annotated.pdf` |
| Figure 5 | Q2–Q3 Van Vleck modes of every TMO6 octahedron, coloured by oxidation state | the `Figure4_vanvleck*` chain, see below; assembled with the axis schematic in `van_vleck/collect.pptx` | `Figure4_vanvleck_oxi_matrix.pdf`, `Figure4_vanvleck_full.pdf` |
| Figure 6 | gravimetric / volumetric energy density vs peak volumetric strain (Pareto map) | `Figure5_setting.py` (all computation) + `Figure5a.py`, `Figure5b.py` (thin wrappers) | `Figure5a_pareto_gravimetric.pdf`, `Figure5b_pareto_volumetric.pdf` |
| Figure 7 | O3-vs-P3 descriptors: (a) slab anisotropy d(O-Na-O)/d(O-TM-O) vs d(O-Na-O); (b) cationic-potential map (Zhao et al. 2020) | `Figure4_setting.py` (geometry of every hull structure) + `Figure4d.py` (a); `Figure4e.py` (b) | `Figure4d_ratio_scatter.pdf`, `Figure4e_cationic_potential.pdf` |

Numbers quoted in the Discussion (measured-vs-predicted voltages of NaxMn0.5Fe0.5O2, energy densities
within the practical Fe limits) are printed by `Discussion_numbers.py`.

## Supporting Information

| SI item | content | script(s) | output |
|---|---|---|---|
| Figure S1, S2 | Na/vacancy ordering diagrams (O3, P3) | drawn by hand: `ordering_diagrams/*.pdf`, `ordering_diagrams/si_ordeing_diagram.pptx` | `SI_Figure1_O3orderings.pdf`, `SI_Figure2_P3orderings.pdf` |
| Figure S3, S4 | convex hulls of the 5 single-TM and 10 mixed systems | `FigureS3S4.py` | `convex_hulls/convexhull_<sys>.pdf`, collected in `SI_Figure3_convexhull_singleTM.pdf`, `SI_Figure4_convexhull_doubleTM.pdf` (`convex_hulls/*_annotated.pdf` carry the ordering labels and duplicate cross-references) |
| Table S1 | experimental vs DFT oxidation states | numbers from `data/doubleTM_stable_oxi_states.csv` | — |
| Figure S5 | Jahn–Teller axial/equatorial bond-length ratio vs x per element | `Figure4_setting.py` + `Figure4b.py` (per system), `Figure4c.py` (all) | `Figure4b_TMO_ratio.pdf`, `Figure4c_ratio_all.pdf` |
| Figure S6, S7 | simulated Cu Kα XRD of every hull structure | `FigureS5S6.py` | `xrd_patterns/xrd_<sys>.pdf`, collected in `SI_Figure5_singleTM_XRD.pdf`, `SI_Figure6_doubleTM_XRD.pdf` |

## The Van Vleck chain (Figure 5) — run in this order

```
Figure4_vanvleck.py            hull structures → one common octahedral frame per structure
                               (z = elongated axis of the most elongated octahedron) → Q1…Q6, rho, phi
                               writes  van_vleck/van_vleck_modes.csv
Figure4_vanvleck_oxi.py        per-site oxidation states from the magnetic moments
                               (single-TM rules + the mixed-TM assignments of
                               data/doubleTM_stable_oxi_states.csv) joined with Q2/Q3
                               writes  data/singleTM_vanvleck_oxi_states.csv, data/doubleTM_vanvleck_oxi_states.csv
Figure4_vanvleck_oxi_polar_each.py   one polar Q2–Q3 panel per system  → van_vleck/Figure4_vanvleck_oxi_polar_<sys>.pdf
Figure4_vanvleck_oxi_matrix.py       the 5×5 lower-triangular matrix     → Figure4_vanvleck_oxi_matrix.pdf (+ _no_legend)
Figure4_vanvleck_oxi_empty_polar.py  empty polar axes for the schematic  → Figure4_vanvleck_oxi_empty_polar.pdf
Figure4_vanvleck_oxi_analysis.py     distortion statistics quoted in the text (counts of distorted /
                               elongated octahedra, parallel neighbours, O atoms with 0/1/2 long bonds)
                               writes  van_vleck/vanvleck_oxi_analysis_{octahedra,structures}.csv
```

`Figure4_vanvleck.py` needs **VanVleckCalculator** (Nagle-Cocco & Dutton 2024), which is not on PyPI:
`git clone https://github.com/lnaglecocco/VanVleckCalculator.git` next to this repository, or set
`VVC_CODE=/path/to/VanVleckCalculator/code`. Note the package's sign convention: a y-elongated
octahedron comes out at φ = 120°, which is why `phi_deg` is compared within a common per-structure frame.

## Earlier / exploratory versions kept for reference (not in the final manuscript)

* `Figure4_vanvleck_plot.py`, `Figure4_vanvleck_polar_each.py` — first Van Vleck figure without oxidation-state colouring (`Figure4_vanvleck_polar.pdf`, `van_vleck/Figure4_vanvleck_polar_<sys>.pdf`).
* `Figure4a.py` — Na-slab thickness vs x (`Figure4a_slab_thickness.pdf`); superseded by Figure 7a.
* `FigureS_redox_ladder_wide.py` — wide layout of Figure 4a.
* `FigureS_JT_orbital_ordering.py` — schematic of cooperative vs local Jahn–Teller orbital ordering.
* `Figure4.pdf`, `Figure2_voltage_caxis_old.pdf`, `Figure1.pdf`, `Figure1_full.pdf`, `FigureS.pdf` — earlier draft versions.

## Assembly files

`collect.pptx` (this folder) and the `collect.pptx` in each sub-folder are the PowerPoint files in which
the per-system PDF panels were arranged into the multi-panel figures.
