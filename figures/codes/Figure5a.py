# Figure 5a: gravimetric energy density (Wh/kg) vs peak volumetric strain (Pareto).
# Shared computation/plotting lives in Figure5_setting.py.

from Figure5_setting import *          # noqa: F401,F403

make_pareto(OUT_GRAV, volumetric=False, figsize=(7.2, 6.0), label_size=23, square=True)
