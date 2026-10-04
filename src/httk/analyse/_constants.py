"""Physical constants and unit factors shared by the analysis kernels."""

from httk.core.units import default_registry

# Exact SI 2019 values: Boltzmann constant and Planck constant over the elementary charge.
KB_EV_PER_K = 1.380649e-23 / 1.602176634e-19
H_EV_PER_THZ = 6.62607015e-34 / 1.602176634e-19 * 1e12
# hbar^2 / m_e in eV angstrom^2 (CODATA 2022 electron mass).
HBAR2_OVER_ME_EV_A2 = 7.619964231073853

# Exact factors between the kernels' eV/angstrom/ps system and the units of the property definitions.
GPA_PER_EV_PER_A3 = float(default_registry().factor("angstrom^-3*eV", "GPa").factor)
M2_PER_S_PER_A2_PER_PS = float(default_registry().factor("angstrom^2*ps^-1", "m^2*s^-1").factor)
JM2_PER_EV_PER_A2 = float(default_registry().factor("angstrom^-2*eV", "J*m^-2").factor)
