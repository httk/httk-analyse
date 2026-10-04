"""Vendored analysis definitions load, match their constants and carry valid units."""

import pytest
from httk.core import load_property_definition
from httk.core.units import default_registry

import httk.analyse.definitions as defs

DERIVATIONS = {"MEAN", "STANDARD_ERROR", "STANDARD_DEVIATION", "RMSE", "MAE", "BIAS", "MAXIMUM_ABSOLUTE_ERROR"}
PROPERTIES = [n for n in defs.__all__ if n.isupper() and n not in DERIVATIONS]


@pytest.mark.parametrize("name", PROPERTIES)
def test_property_loads_with_matching_id_and_unit(name):
    iri = getattr(defs, name)
    d = load_property_definition(iri)
    assert d.definition_id == iri
    if d.unit not in (None, "", "inapplicable", "dimensionless"):
        default_registry().dimension(d.unit)


@pytest.mark.parametrize("name", sorted(DERIVATIONS))
def test_derivations_are_not_properties(name):
    iri = getattr(defs, name)
    assert iri.startswith("https://schemas.httk.org/defs/v0.1/derivations/")
    with pytest.raises(ValueError):
        load_property_definition(iri)
