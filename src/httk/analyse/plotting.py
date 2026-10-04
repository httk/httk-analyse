"""Matplotlib figures for analysis results with explicit physical axes.

Axes of quantities with a property definition are labelled with the
definition's title and unit.
"""

from collections.abc import Mapping
from functools import cache
from importlib import import_module
from typing import Any

import numpy as np
from httk.core import definition_ids, load_property_definition

from . import definitions as defs
from .matsci.dynamics import TensorSeries
from .matsci.energetics import ChemicalPotentialRegion, ConvergenceTable
from .matsci.eos import BirchMurnaghanFit
from .matsci.eos_models import EOSFit
from .matsci.phonons import HarmonicThermodynamics
from .matsci.structure import RadialDistribution
from .matsci.transport import TransportResult
from .matsci.validation import PropertyParity

__all__ = [
    "plot_chemical_potential_slice",
    "plot_convergence",
    "plot_eos",
    "plot_msd",
    "plot_parity",
    "plot_phonons",
    "plot_rdf",
    "plot_transport",
]


def plot_eos(result: BirchMurnaghanFit | EOSFit) -> tuple[Any, Any]:
    """Plot sampled energies, fitted EOS and residuals on a shared volume axis.

    :param result: EOS fit with input volumes, energies and residuals.
    :return: Matplotlib figure and two axes; the caller owns their lifecycle.
    """
    plt = import_module("matplotlib.pyplot")
    figure, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    volumes = np.linspace(min(result.volumes), max(result.volumes), 200)
    axes[0].plot(result.volumes, result.energies, "o", label="samples")
    axes[0].plot(volumes, [result.energy(float(v)) for v in volumes], label="fit")
    axes[0].set_ylabel(_label(definition_ids.TOTAL_ENERGY))
    axes[0].legend()
    axes[1].plot(result.volumes, result.residuals, "o")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set(xlabel=_label(definition_ids.VOLUME), ylabel="Observed − fitted (eV)")
    return figure, axes


def plot_convergence(result: ConvergenceTable) -> tuple[Any, Any]:
    """Plot signed energy differences in caller-supplied parameter order.

    :param result: Convergence data with explicit reference index.
    :return: Matplotlib figure and axis.
    """
    figure, axis = _axis()
    axis.plot(range(len(result.parameters)), result.differences_per_atom, "o-")
    axis.set_xticks(range(len(result.parameters)), [str(v) for v in result.parameters])
    axis.set(xlabel="Calculation parameter", ylabel="Energy − reference (eV/atom)")
    return figure, axis


def plot_rdf(result: RadialDistribution) -> tuple[Any, Any]:
    """Plot the normalized radial distribution at shell centers.

    :param result: RDF with its declared normalization.
    :return: Matplotlib figure and axis.
    """
    figure, axis = _axis()
    axis.plot(result.centers, result.g)
    axis.set(xlabel="Radius (angstrom)", ylabel="g(r)")
    return figure, axis


def plot_msd(result: TensorSeries) -> tuple[Any, Any]:
    """Plot an MSD tensor trace without fitting or selecting a diffusive window.

    :param result: Tensor series returned by mean_squared_displacement.
    :return: Matplotlib figure and axis.
    :raises ValueError: If the result is not an MSD series.
    """
    if result.kind != "msd":
        raise ValueError("plot_msd requires an MSD tensor series")
    figure, axis = _axis()
    axis.plot(result.times, result.trace)
    axis.set(xlabel="Lag time (ps)", ylabel="Mean squared displacement (angstrom²)")
    return figure, axis


def plot_transport(result: TransportResult) -> tuple[Any, Any]:
    """Plot all running Green–Kubo components and their isotropic mean.

    :param result: Thermal conductivity or shear viscosity running integrals.
    :return: Matplotlib figure and axis.
    """
    figure, axis = _axis()
    for index, label in enumerate(result.components):
        axis.plot(result.times, np.asarray(result.integrals)[:, index], label=label, alpha=0.6)
    axis.plot(result.times, result.isotropic, color="black", linewidth=2, label="isotropic mean")
    definition = defs.THERMAL_CONDUCTIVITY if result.quantity == "thermal_conductivity" else defs.SHEAR_VISCOSITY
    axis.set(xlabel="Integration time (ps)", ylabel=_label(definition))
    axis.legend()
    return figure, axis


def plot_phonons(result: HarmonicThermodynamics) -> tuple[Any, Any]:
    """Plot harmonic free energy, entropy and heat capacity with separate units.

    :param result: Harmonic thermodynamics in its input mode normalization.
    :return: Matplotlib figure and three axes.
    """
    plt = import_module("matplotlib.pyplot")
    figure, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    for axis, values, label in zip(
        axes,
        (result.free_energy, result.entropy, result.heat_capacity),
        (defs.HELMHOLTZ_FREE_ENERGY, defs.VIBRATIONAL_ENTROPY, defs.VIBRATIONAL_HEAT_CAPACITY),
        strict=True,
    ):
        axis.plot(result.temperatures, values)
        axis.set_ylabel(_label(label))
    axes[-1].set_xlabel(_label(definition_ids.TEMPERATURE))
    return figure, axes


def plot_parity(result: PropertyParity) -> tuple[Any, Any]:
    """Plot paired scalar predictions with a unity reference line.

    :param result: Matched property pairs; axes use its definition, or no unit when it has none.
    :return: Matplotlib figure and axis.
    """
    figure, axis = _axis()
    axis.scatter(result.reference, result.predicted)
    limits = [min(*result.reference, *result.predicted), max(*result.reference, *result.predicted)]
    axis.plot(limits, limits, color="grey", linestyle="--")
    quantity = "" if result.definition is None else f": {_label(result.definition)}"
    axis.set(xlabel=f"Reference{quantity}", ylabel=f"Predicted{quantity}", aspect="equal")
    return figure, axis


@cache
def _label(iri: str) -> str:
    definition = load_property_definition(iri)
    return f"{definition.title} ({definition.unit})"


def _axis() -> tuple[Any, Any]:
    return import_module("matplotlib.pyplot").subplots(constrained_layout=True)


def plot_chemical_potential_slice(
    region: ChemicalPotentialRegion,
    *,
    x_element: str,
    y_element: str,
    dependent_element: str,
    fixed: Mapping[str, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    resolution: int = 201,
) -> tuple[Any, Any]:
    """Plot a sampled two-dimensional slice of chemical-potential constraints.

    The host equality eliminates one dependent potential. All remaining
    elements must have explicit fixed potentials. Shading is a finite grid
    visualization, not an exact polygon or a change to the stored constraints.

    :param region: Host equality and competing-phase inequalities.
    :param x_element: Element on the horizontal axis.
    :param y_element: Element on the vertical axis.
    :param dependent_element: Element eliminated using the host equality.
    :param fixed: Potentials for every other element, in eV/atom.
    :param x_limits: Increasing horizontal bounds in eV/atom.
    :param y_limits: Increasing vertical bounds in eV/atom.
    :param resolution: Grid points per axis, at least two.
    :return: Matplotlib figure and axis containing the feasible sampled region.
    :raises ValueError: If elements, bounds, resolution or fixed potentials are invalid.
    """
    chosen = (x_element, y_element, dependent_element)
    if len(set(chosen)) != 3 or any(element not in region.elements for element in chosen):
        raise ValueError("three distinct elements from the region are required")
    if set(fixed) != set(region.elements) - set(chosen):
        raise ValueError("fixed potentials must cover exactly the other elements")
    bounds = np.asarray([x_limits, y_limits], dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("limits must be finite increasing pairs")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer at least two")
    if not all(np.isfinite(value) for value in fixed.values()):
        raise ValueError("fixed potentials must be finite")
    ix, iy, idep = (region.elements.index(element) for element in chosen)
    host = np.asarray(region.host_coefficients)
    if host[idep] == 0:
        raise ValueError("dependent element must have a nonzero host coefficient")
    x, y = np.meshgrid(np.linspace(*x_limits, resolution), np.linspace(*y_limits, resolution))
    potentials = np.empty((*x.shape, len(host)))
    potentials[:, :, ix], potentials[:, :, iy] = x, y
    constant = 0.0
    for element, value in fixed.items():
        index = region.elements.index(element)
        potentials[:, :, index] = value
        constant += host[index] * value
    potentials[:, :, idep] = (region.host_energy - host[ix] * x - host[iy] * y - constant) / host[idep]
    feasible = np.ones(x.shape, dtype=bool)
    for coefficients, energy in zip(region.competing_coefficients, region.competing_energies, strict=True):
        feasible &= potentials @ np.asarray(coefficients) <= energy
    figure, axis = _axis()
    axis.contourf(x, y, feasible.astype(float), levels=[-0.5, 0.5, 1.5], colors=["white", "#70b7ab"])
    axis.set(xlabel=f"mu({x_element}) (eV/atom)", ylabel=f"mu({y_element}) (eV/atom)")
    return figure, axis
