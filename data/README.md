# data/ — data dictionary

All tabulated inputs and results. Paths are relative to this directory. "Produced by" names the
script/notebook that wrote the file; "Used by" names the consumers inside this repository.

Common key columns (see the root README for the naming scheme):
`phase` (O1_orderings / O3_orderings / P3_orderings), `ordering`, `tm` or `tm_pair` + `elec`,
`xNa` (Na per TMO2 formula unit), structure key `phase-ordering-tm[_pair-elec]`.

---

## 1. Na/vacancy ordering library (DFT inputs)

| directory | contents |
|---|---|
| `O3_orderings/` | 17 Toumar orderings `AJT_a…r` (no `AJT_j`), 4 Kaufman orderings `JLK_b, JLK_c, JLK_theta_4, JLK_theta_5`, experimental `JMAS_a, JMAS_b, JMAS_e` (NaxNiO2, Steele), `XC_a, XC_b` (NaxMnO2, Chen 2018), `XL_a` (Na5/8MnO2, Li 2014), `CD_a` (Na1/2VO2, Didier 2012), hosts `O3_full`, `O3_empty` (+ `_conventional` 1-layer cells), and the 7×7×1 NaCoO2 host supercell `O3_NaCoO2_POSCAR_771.vasp` from which the AJT/JLK cells were cut |
| `P3_orderings/` | Kaufman P3 orderings `JLK_delta, JLK_eta, JLK_zeta, JLK_zeta2, JLK_zeta3, JLK_zeta6`, experimental `JMAS_c, JMAS_d`, hosts `P3_full`, `P3_empty`, and the 7×7×1 host `P3_NaCoO2_771.vasp` |
| `O1_orderings/` | `O1_empty` host only (O1 appears only at x = 0) |
| `experiments/orderings/` | the refined CIF files the JMAS / XC / CD orderings were converted from (`notebooks/misc.ipynb`) |

File conventions:

* `NAME_w_vac.vasp` — the ordering as used: occupied Na sites are **Na**, vacant Na sites are **K**
  (vacancy marker, removed by every script before calculation). All sites are on the NaCoO2 host;
  Co is substituted by the target TM downstream.
* `NAME.vasp` without `_w_vac` — same supercell with every Na site filled (reference for cutting).
* Experimental orderings (`JMAS_*`, `XC_*`, `XL_a`, `CD_a`) contain real vacancies and no K.
* Duplicates imported from different sources: XC_a = JMAS_a = AJT_d, JMAS_c = JLK_zeta, AJT_g = CD_a, JMAS_b = AJT_f.

## 2. Digitized experimental voltage curves — `experiments/*.csv`

Two columns, no header: `x` (either Na content or capacity in mAh/g — see `is_capacity` in
`figures/codes/Figure2.py`) and `voltage (V vs Na/Na+)`. Named `<journal><year>_<compound>.csv`:

| file | system | source |
|---|---|---|
| `EES2015_NTO.csv` | NaxTiO2 | Wu et al., Energy Environ. Sci. 2015 |
| `AFM2018_NMO_charge.csv`, `AFM2018_NMO_discharge.csv` | NaxMnO2 | Chen et al., Adv. Funct. Mater. 2018 |
| `AEM2018_NFO.csv`, `AEM2018_NCO.csv` | NaxFeO2, NaxCoO2 | Kubota et al., Adv. Energy Mater. 2018 |
| `AFM2016_NCO.csv`, `AFM2016_NFC.csv` | NaxCoO2, NaxFe0.5Co0.5O2 | Kubota et al., Adv. Funct. Mater. 2016 |
| `ChemRxiv_NNO.csv` | NaxNiO2 | Steele et al., J. Am. Chem. Soc. 2026 (ChemRxiv preprint) |
| `SSI1981_NCO.csv` | NaxCoO2 | Delmas et al., Solid State Ionics 1981 |
| `JPC2015_NCrO.csv` | NaxCrO2 | Kubota et al., J. Phys. Chem. C 2015 (not in the paper) |
| `IC2012_NVO.csv` | NaxVO2 | Didier et al., Inorg. Chem. 2012 (not in the paper) |
| `NM2012_NMF.csv` | NaxMn0.5Fe0.5O2 | Yabuuchi et al., Nat. Mater. 2012 |
| `JMCA2017_NFN.csv`, `JMCA2017_NCN.csv` | NaxFe0.5Ni0.5O2, NaxCo0.5Ni0.5O2 | Vassilaras et al., J. Mater. Chem. A 2017 |
| `IC2012_NMN.csv` | NaxMn0.5Ni0.5O2 | Komaba et al., Inorg. Chem. 2012 |
| `CM2016_NTN.csv` | NaxTi0.5Ni0.5O2 | Nanba et al., Chem. Mater. 2016 |

Used by `figures/codes/Figure2.py`, `Discussion_numbers.py`, and the voltage notebooks.

## 3. MACE pre-screening candidates — `mace_screening/`

### `singleTM/`
* `ordering_staging/<phase>/<ordering>/group_NNN/POSCAR` + `group_meta.json` — every symmetry-distinct
  three-layer stacking of each ordering (231 groups), from `notebooks/Staging_layers_noTMMix.ipynb`.
  K marks vacancies.
* `ordering_staging_output/<phase>/<ordering>/group_NNN/<TM>/final.POSCAR` + `summary.json` — MACE-OMAT-0
  relaxed structure (cell + ions) for each group × TM, from `scripts/mace_ordering_calculation_cli.py`;
  `summary.csv` collects all 1616 relaxations (energy, volume, fmax, steps).
* `minima_summary.csv` — the lowest-energy group chosen per (phase, ordering, TM) = the DFT starting
  structures (`scripts/dft_prep_from_mace_singleTM.py`).

### `doubleTM/ordering_doubleTM/<phase>/<ordering>/<TM1_TM2>/`
From `scripts/util_tmmix_tmstaging.py` → `mace_eval_tmmix_tmstaging.py` → `util_tmmix_nastaging.py` → `mace_eval_tmmix_nastaging.py`.
Only the two Ewald-lowest in-plane TM configurations (`elec_0`, `elec_1`) that were sent to DFT are kept.
* `target_layer_data.json` — in-plane translation vectors of the TM sublattice (per layer).
* `elec_N/step1_meta.json`, `POSCAR_step1` — reference TM layer mixed 1:1 (Ewald rank N, cell doubled if the layer has an odd number of TM sites).
* `elec_N/POSCAR_TMstaging`, `TM_staging_mace_summary.json` — TM-layer stacking chosen by single-point MACE energy.
* `elec_N/POSCAR_lowestEconfig.vasp`, `Na_staging_mace_summary.json` — Na-layer stacking chosen by single-point MACE energy = **the DFT input structure**.

## 4. DFT results

### 4.1 Full run tables (one row per VASP relaxation)

| file | rows | content |
|---|---|---|
| `singleTM_tighter_conv.csv` | 287 | **hull dataset, single TM** (Ti, V, Cr, Mn, Fe, Co, Ni). Tight settings: ENCUT 680, KSPACING 0.3, EDIFF 1e-6/atom, EDIFFG −0.02, D3(BJ). |
| `doubleTM_dft_tighter.csv` | 820 | **hull dataset, mixed TM** (10 pairs × elec_0/elec_1), same settings. |
| `singleTM_dft_summary.csv`, `singleTM_w_exp.csv` | 224 / 280 | first-pass MPRelaxSet (520 eV) relaxations; `_w_exp` adds the experimental orderings |
| `singleTM_w_exp_2nd_relax.csv` | 287 | second MPRelaxSet relaxation of the first pass (used as the MP-compatible energy for ehull) |
| `doubleTM_dft_summary.csv`, `doubleTM_dft_2nd_relax_summary.csv` | 820 / 820 | mixed-TM analogues of the two lines above |

Columns (from `scripts/dft_collect_from_*.py`): `run_dir, relpath, phase, ordering, tm|tm_pair, elec,
meta_* (MACE provenance, single TM only), converged_electronic, converged_ionic, converged, nsteps_ionic,
energy_eV, energy_per_atom_eV, volume_A3, lattice (pymatgen Lattice dict), natoms, formula (reduced!),
efermi_eV, total_mag, status, error_msg, n_custodian_errors, custodian_handlers, custodian_error_types,
custodian_corrections, incar_nsw, ionic_hit_nsw_limit, nonconv_fix_* (bookkeeping of NSW bumps)`.
Energies per formula unit are obtained as `energy_eV / n_TMO2` with `n_TMO2` from `natoms` and the
composition; the notebooks and `FigureS3S4.py` show the exact normalization.

### 4.2 Relaxed structures

| file | entries | key |
|---|---|---|
| `singleTM_structures.json` | 287 | `phase-ordering-tm` |
| `doubleTM_structures.json` | 820 | `phase-ordering-tm_pair-elec` |
| `structures.json` | 1107 | union of the two (kept because several scripts read it directly) |

Values are `pymatgen.core.Structure.as_dict()` of the final CONTCAR of the tight relaxation
(`scripts/dft_collect_CONTCAR_from_*.py`). Load with
`Structure.from_dict(json.load(open("singleTM_structures.json"))[key])`.

### 4.3 Magnetic moments (from OUTCAR)

`singleTM_magmom.csv` (`phase, ordering, tm, tm_magmom_dict`) and `doubleTM_magmom.csv`
(`…, tm_pair, tm_ordering, tm_magmom_dict`); `tm_magmom_dict` is a Python-literal dict
`{'Co': [m1, m2, …]}` in site order (parse with `ast.literal_eval`). From `scripts/dft_collect_magmom_*.py` (the `--run-type` option selects relax / 2nd_relax / static sub-runs; the tables here come from the tight relaxations).

### 4.4 Derived tables (from the notebooks)

| file | produced by | content |
|---|---|---|
| `singleTM_hull_points.csv`, `doubleTM_hull_points.csv` | `notebooks/dft_structure_and_thermo_analysis.ipynb` | the hull-stable compositions only: `tm|tm_pair, hull_order, phase, ordering, [elec], xNa, E_TMO2_eV, formula`. **Authoritative list of ground states** used by Figures 2–7. |
| `singleTM_hull_mp_ehull.csv`, `doubleTM_hull_mp_ehull.csv` | same notebook, step 2 (needs `MP_API_KEY`) | hull points + energy above the Materials Project hull (`ehull`, eV/atom, with MP2020 corrections), decomposition products, and the r1/r2/best energies used. Colours of Figure 3. |
| `doubleTM_stable_oxi_states.csv` | `notebooks/dft_oxidation_analysis.ipynb` | per hull point of the mixed systems: per-site moments, per-site oxidation states assigned from the moments (`ox_tm1_list`, `ox_tm2_list`, charge-neutrality resolved), per-element averages (`avg_ox_tm1`, `avg_ox_tm2`). Numbers of Figure 3, Table S1, Figure 4. |
| `singleTM_vanvleck_oxi_states.csv`, `doubleTM_vanvleck_oxi_states.csv` | `figures/codes/Figure4_vanvleck_oxi.py` | one row per TMO6 octahedron of every hull structure: `site_index, element, magmom_muB, possible_charges, charge, ion, note (assigned / n/a), Q2_A, Q3_A, rho_A, phi_deg`. Data of Figure 5. |

The Van Vleck mode table itself (`Q1…Q6, rho, phi, eta` per octahedron) is
`../figures/van_vleck/van_vleck_modes.csv` (from `Figure4_vanvleck.py`).
