# Numbers quoted in the Discussion section of the DFT paper. Nothing is drawn and no file is written.
#
# 1. NaxMn0.5Fe0.5O2: how far the measured charge curve of Figure 2 (O3 sample of Yabuuchi et al.,
#    Nat. Mater. 2012) is located above the predicted voltage.
# 2. Pareto figure (Figure 6 of the paper): energy density and peak volumetric strain of the Fe-based
#    systems at the end of their curves and at the stable composition that is closest to the experimental
#    limit of the reversible cycling, together with NaxMn0.5Ni0.5O2 and NaxTi0.5Ni0.5O2.
#
# Data: data/doubleTM_hull_points.csv, data/experiments/NM2012_NMF.csv, and the curves of Figure5_setting.py.
# Run:  conda run -n research python Discussion_numbers.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pymatgen.core.composition import Composition

import Figure5_setting as pareto

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
E_NA = -1.31                 # eV/atom, bcc Na (same reference as in Figure2.py)
FARADAY = 96485.33212        # C/mol

# Na content down to which the experiments cycle the material reversibly (Results section of the paper)
REVERSIBLE_LIMIT = {
    "Fe": 0.50,              # NaxFeO2: Kubota et al. 2018, Lee et al. 2015
    "Fe-Co": 0.30,           # NaxFe0.5Co0.5O2: Kubota et al. 2016
    "Fe-Ni": 0.34,           # NaxFe0.5Ni0.5O2: Vassilaras et al. 2017
}
COMPARED_WITH = ["Mn-Ni", "Ti-Ni", "Co"]      # systems printed next to the Fe-based ones


# ---------------------------------------------------------------------------------------------------
# 1. NaxMn0.5Fe0.5O2: measured charge curve against the predicted voltage
# ---------------------------------------------------------------------------------------------------
print("1. NaxMn0.5Fe0.5O2, measured charge curve (Figure 2) against the predicted voltage")

hull = pd.read_csv(DATA_DIR / "doubleTM_hull_points.csv")
hull = hull[hull.tm_pair == "Mn_Fe"].sort_values("xNa")
x_hull, e_hull = hull.xNa.to_numpy(), hull.E_TMO2_eV.to_numpy()
# one voltage step between two neighboring stable compositions: (x_low, x_high, voltage)
steps = [(x_hull[i], x_hull[i + 1], -((e_hull[i + 1] - e_hull[i]) / (x_hull[i + 1] - x_hull[i]) - E_NA))
         for i in range(len(x_hull) - 1)]


def predicted_voltage(x):
    return next(v for x_low, x_high, v in steps if x_low - 1e-9 <= x <= x_high + 1e-9)


measured = pd.read_csv(DATA_DIR / "experiments" / "NM2012_NMF.csv", header=None, names=["capacity", "voltage"])
capacity_one_na = FARADAY / 3.6 / Composition("NaMn0.5Fe0.5O2").weight          # mAh/g for one Na
measured["xNa"] = 1 - measured.capacity / capacity_one_na
measured = measured.sort_values("xNa")
x_min, x_max = measured.xNa.min(), min(measured.xNa.max(), 1.0)
print(f"   measured curve: x = {x_max:.2f} to {x_min:.2f}, {measured.voltage.min():.2f} to {measured.voltage.max():.2f} V")

for x_low, x_high, v in reversed(steps):
    part = measured[(measured.xNa >= x_low) & (measured.xNa <= x_high)]
    text = f"measured {part.voltage.mean():.2f} V on average" if len(part) else "not measured"
    print(f"   x = {x_high:.3f} to {x_low:.3f}: predicted {v:.2f} V, {text}")

grid = np.linspace(x_min + 1e-4, x_max - 1e-4, 2000)
v_measured = np.interp(grid, measured.xNa, measured.voltage)
v_predicted = np.array([predicted_voltage(x) for x in grid])
difference = v_measured - v_predicted
print(f"   over the measured range: measured {v_measured.mean():.2f} V, predicted {v_predicted.mean():.2f} V, "
      f"measured minus predicted {difference.mean():+.2f} V on average, "
      f"largest {difference.max():+.2f} V at x = {grid[difference.argmax()]:.2f}")


# ---------------------------------------------------------------------------------------------------
# 2. Pareto figure: end of the curve against the limit of the reversible cycling
# ---------------------------------------------------------------------------------------------------
print("\n2. Pareto figure: energy density (Wh/kg) and peak volumetric strain (%)")

curves = {label: trace for label, _, trace in pareto.singles + pareto.doubles}
for label, limit in REVERSIBLE_LIMIT.items():
    strain, energy, x, _ = curves[label]
    closest = int(np.argmin(np.abs(x - limit)))          # stable composition closest to the limit
    print(f"   {label:6s} end of the curve x = {x[-1]:.3f}: {energy[-1]:.0f} Wh/kg at {strain[-1]:.2f} % | "
          f"limit x = {limit:.2f}, closest stable composition x = {x[closest]:.3f}: "
          f"{energy[closest]:.0f} Wh/kg at {strain[closest]:.2f} %")
for label in COMPARED_WITH:
    strain, energy, x, _ = curves[label]
    points = ", ".join(f"x = {xi:.3f}: {e:.0f} Wh/kg at {s:.2f} %" for xi, e, s in zip(x[1:], energy[1:], strain[1:]))
    print(f"   {label:6s} {points}")
