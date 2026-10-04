"""Equilibrium fluctuation responses for explicitly selected ensembles."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from httk.core import definition_ids

from .. import definitions as defs
from .._constants import GPA_PER_EV_PER_A3, KB_EV_PER_K
from ..definitions import _PROPERTY_BY_NAME, BoundValue, FieldBinding, _reject_selection
from .dynamics import _array

__all__ = ["EquilibriumResponse", "equilibrium_response"]


@dataclass(frozen=True, slots=True)
class EquilibriumResponse:
    """Fluctuation estimates with optional independent-block error estimates.

    :param ensemble: The supplied equilibrium ensemble.
    :param names: Property definition names in value order; values are in the units of those definitions.
    :param values: Estimates from all retained samples, using population moments.
    :param temperature: Supplied equilibrium temperature in K.
    :param block_values: Per-block property estimates, when requested.
    :param standard_errors: Sample SD of block estimates divided by sqrt(block count), or absent.
    :param used_samples: Number of retained samples.
    :param dropped_samples: Explicitly discarded tail samples.
    """

    ensemble: Literal["NVT", "NPT"]
    names: tuple[str, ...]
    values: tuple[float, ...]
    temperature: float
    block_values: tuple[tuple[float, ...], ...]
    standard_errors: tuple[float, ...] | None
    used_samples: int
    dropped_samples: int

    def __post_init__(self) -> None:
        """Copy numeric values and metadata into immutable tuples."""
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "values", tuple(float(v) for v in _array(self.values, "values")))
        object.__setattr__(self, "block_values", tuple(tuple(float(v) for v in row) for row in self.block_values))
        if self.standard_errors is not None:
            object.__setattr__(self, "standard_errors", tuple(float(v) for v in self.standard_errors))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind each named response, the temperature and any block standard errors."""
        _reject_selection(self, selection)
        unknown = [name for name in self.names if name not in _PROPERTY_BY_NAME]
        if unknown:
            raise ValueError(f"names are not analysis property definitions: {', '.join(unknown)}")
        bound = [BoundValue(name, FieldBinding(_PROPERTY_BY_NAME[name]), v) for name, v in zip(self.names, self.values)]
        bound.append(BoundValue("temperature", FieldBinding(definition_ids.TEMPERATURE), float(self.temperature)))
        if self.standard_errors is not None:
            bound += (
                BoundValue(f"standard_errors.{name}", FieldBinding(_PROPERTY_BY_NAME[name], defs.STANDARD_ERROR), v)
                for name, v in zip(self.names, self.standard_errors)
            )
        return tuple(bound)


def equilibrium_response(
    *,
    temperature: float,
    ensemble: Literal["NVT", "NPT"],
    energies: Sequence[float] | None = None,
    enthalpies: Sequence[float] | None = None,
    volumes: Sequence[float] | None = None,
    block_size: int | None = None,
    remainder: Literal["raise", "drop"] = "raise",
) -> EquilibriumResponse:
    """Estimate canonical or isothermal-isobaric fluctuation responses.

    NVT uses total E and gives Cv=Var(E)/(kB*T²). NPT requires total H and V,
    giving Cp=Var(H)/(kB*T²), kappa=Var(V)/(kB*T*mean(V)), and volumetric
    alpha=Cov(V,H)/(kB*T²*mean(V)). Population moments use ddof=0. All samples
    receive equal weight and must represent equilibrium at the supplied T.
    Thermostat/barostat parameters or a linear drift check do not establish
    correct ensemble sampling. These formulas do not apply to NVE energy data.

    E, H and V must be whole-system totals of the simulated cell, never per-atom
    or otherwise normalized values. Per-atom inputs (for example LAMMPS
    ``thermo_modify norm yes``) divide Cv and Cp by N² and kappa and alpha by N;
    divide the whole-system results by N afterwards instead. NPT enthalpy is
    H = E_total + P_ext*V with E_total including kinetic energy and P_ext the
    fixed barostat set-point pressure. The LAMMPS ``enthalpy`` thermo keyword
    uses the instantaneous pressure instead and biases Cp and alpha.

    Block standard errors assume independent blocks; choose block length much
    longer than the correlation time tau and check block-size sensitivity. Each
    block uses population moments about its own mean, so short blocks bias
    variances low by a factor (n_b - 1)/n_b plus about 2*tau/n_b, with tau the
    integrated correlation time in samples and n_b the block length in samples. The full
    estimate need not equal the mean block estimate.

    :param temperature: Positive fixed equilibrium temperature in K.
    :param ensemble: Explicit NVT or NPT ensemble.
    :param energies: NVT whole-system total energies in eV; required for NVT only.
    :param enthalpies: NPT whole-system total enthalpies in eV; required for NPT only.
    :param volumes: NPT whole-system volumes in angstrom³, matching enthalpies.
    :param block_size: Optional samples per block, at least two; requires two complete blocks.
    :param remainder: Raise on an incomplete block or explicitly drop the tail.
    :return: Values named after property definitions: extensive ``heat_capacity_constant_volume`` or
        ``heat_capacity_constant_pressure`` in eV/K, ``isothermal_compressibility`` in GPa⁻¹ and
        ``volumetric_thermal_expansion`` in K⁻¹ (the NPT pair).
    :raises ValueError: If ensemble inputs, temperature, blocks or derived values are invalid.
    """
    temp = _array(temperature, "temperature")
    if temp.ndim != 0 or float(temp) <= 0:
        raise ValueError("temperature must be a positive finite scalar")
    t = float(temp)
    names: tuple[str, ...]
    if ensemble == "NVT":
        if energies is None or enthalpies is not None or volumes is not None:
            raise ValueError("NVT requires energies only, without enthalpies or volumes")
        values = _array(energies, "energies")
        names = ("heat_capacity_constant_volume",)
        volume = None
    elif ensemble == "NPT":
        if enthalpies is None or volumes is None or energies is not None:
            raise ValueError("NPT requires enthalpies and volumes, without energies")
        values = _array(enthalpies, "enthalpies")
        volume = _array(volumes, "volumes")
        if volume.shape != values.shape or np.any(volume <= 0):
            raise ValueError("volumes must be positive and match enthalpies")
        names = ("heat_capacity_constant_pressure", "isothermal_compressibility", "volumetric_thermal_expansion")
    else:
        raise ValueError("ensemble must be NVT or NPT")
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two scalar equilibrium samples are required")
    if remainder not in ("raise", "drop"):
        raise ValueError("remainder must be raise or drop")
    used, dropped = len(values), 0
    blocks: tuple[tuple[float, ...], ...] = ()
    errors = None
    if block_size is not None:
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 2:
            raise ValueError("block_size must be an integer at least two")
        block_size = int(block_size)
        count, dropped = divmod(len(values), block_size)
        if count < 2 or (dropped and remainder == "raise"):
            raise ValueError("two complete blocks are required; incomplete tails need remainder='drop'")
        used = count * block_size
        blocks = tuple(
            _response(values[i : i + block_size], None if volume is None else volume[i : i + block_size], t)
            for i in range(0, used, block_size)
        )
        error_values = np.std(np.asarray(blocks), axis=0, ddof=1) / math.sqrt(count)
        if not np.isfinite(error_values).all():
            raise ValueError("block errors are not finite")
        errors = tuple(float(v) for v in error_values)
    estimate = _response(values[:used], None if volume is None else volume[:used], t)
    return EquilibriumResponse(ensemble, names, estimate, t, blocks, errors, used, dropped)


def _response(energy: np.ndarray, volume: np.ndarray | None, temperature: float) -> tuple[float, ...]:
    centered_energy = energy - energy[0]
    centered_energy -= centered_energy.mean()
    capacity = float(np.mean(centered_energy**2) / (KB_EV_PER_K * temperature**2))
    result: tuple[float, ...]
    if volume is None:
        result = (capacity,)
    else:
        mean_volume = float(volume.mean())
        centered_volume = volume - volume[0]
        centered_volume -= centered_volume.mean()
        compressibility = float(
            np.mean(centered_volume**2) / (KB_EV_PER_K * temperature * mean_volume) / GPA_PER_EV_PER_A3
        )
        expansion = float(np.mean(centered_volume * centered_energy) / (KB_EV_PER_K * temperature**2 * mean_volume))
        result = capacity, compressibility, expansion
    if not all(math.isfinite(value) for value in result):
        raise ValueError("fluctuation responses are not finite")
    return result
