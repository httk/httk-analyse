"""Provenance envelopes retain file identities and immutable numerical results."""

import hashlib
import json

import pytest
from httk.core import RunEdge
from httk.core.storage import content_id

from httk.analyse import definitions as defs
from httk.analyse.matsci.phonons import harmonic_thermodynamics
from httk.analyse.matsci.transport import thermal_conductivity
from httk.analyse.records import bound_values, records
from httk.analyse.summary import AnalysisSummary, analysis_summary


def test_summary_roundtrip_hash_selection_and_immutability(tmp_path):
    source = tmp_path / 'modes.csv'
    source.write_text('frequency_THz\n2\n')
    result = harmonic_thermodynamics([2], [0, 300])
    summary = analysis_summary(
        result,
        algorithm='harmonic_thermodynamics',
        units={'retained_mode_weight': 'dimensionless', 'cutoff_frequency': 'THz'},
        parameters={'zero_modes': 'raise'},
        selection={'rows': [0]},
        sources=[source],
        assumptions=['one mode per input cell'],
    )
    value = summary.value
    assert value['sources'][0]['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert value['result']['free_energy'] == list(result.free_energy)
    value['result']['free_energy'][0] = 999
    assert summary.value['result']['free_energy'][0] != 999
    output = tmp_path / 'result.json'
    summary.write(output)
    assert AnalysisSummary(output.read_text()) == summary
    assert json.loads(output.read_text())['selection'] == {'rows': [0]}
    fields = value['fields']
    assert fields['zero_point_energy'] == {
        'definition': defs.ZERO_POINT_ENERGY,
        'derivation': None,
        'content_id': content_id(records(result)[0]),
        'value': result.zero_point_energy,
    }
    assert fields['vibrational_thermodynamics']['definition'] == defs.VIBRATIONAL_THERMODYNAMICS
    assert fields['vibrational_thermodynamics']['value']['helmholtz_free_energies'] == list(result.free_energy)
    assert fields['cutoff_frequency'] == {
        'unit': 'THz',
        'unit_definitions': [
            'https://schemas.optimade.org/defs/v1.2/prefixes/si/tera',
            'https://schemas.optimade.org/defs/v1.2/units/si/general/hertz',
        ],
    }
    assert {b.field for b in bound_values(result)} <= set(fields)


def _summary(result, **kwargs):
    return analysis_summary(result, algorithm='x', parameters={}, selection={}, sources=[], assumptions=[], **kwargs)


def test_summary_units_are_only_for_unbound_fields():
    result = harmonic_thermodynamics([2], [0, 300])
    with pytest.raises(ValueError, match='property definition'):
        _summary(result, units={'vibrational_thermodynamics': 'eV'})
    with pytest.raises(ValueError):
        _summary(result, units={'retained_mode_weight': 'eV/angstrom'})
    series = _summary({'edges': [1.0, 2.0]}, units={'edges': 'angstrom'}).value['fields']
    assert series == {
        'edges': {
            'unit': 'angstrom',
            'unit_definitions': ['https://schemas.optimade.org/defs/v1.2/units/si/general/angstrom'],
        }
    }
    with pytest.raises(TypeError):
        _summary({'edges': [1.0]}, lag_index=1)


def test_summary_bound_selection_matches_records():
    result = thermal_conductivity(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], 1, temperature=300, volume=10, max_lag=2
    )
    series = _summary(result).value['fields']
    assert set(series) == {'thermal_conductivity_running_integral', 'temperature', 'volume'}
    fields = _summary(result, lag_index=2).value['fields']
    assert fields['thermal_conductivity_running_integral'] == series['thermal_conductivity_running_integral']
    assert fields['isotropic[2]']['definition'] == defs.THERMAL_CONDUCTIVITY
    assert fields['isotropic[2]']['value'] == result.isotropic[2]  # binding names are not result paths
    assert fields['integrals[2]']['value'] == [list(result.integrals[2][i : i + 3]) for i in (0, 3, 6)]
    assert fields['integrals[2]']['definition'] == defs.THERMAL_CONDUCTIVITY_TENSOR
    with pytest.raises(TypeError):
        _summary(result, lag_index=2, plateau=1)


def test_summary_rejects_nonfinite_and_encodes_complex(tmp_path):
    kwargs = {
        'algorithm': 'scattering',
        'units': {'values': 'dimensionless'},
        'parameters': {},
        'selection': {},
        'sources': [],
        'assumptions': [],
    }
    result = analysis_summary({'values': [1 + 2j]}, **kwargs)
    assert result.value['result']['values'] == [{'real': 1, 'imaginary': 2}]
    with pytest.raises(ValueError):
        analysis_summary({'values': [float('nan')]}, **kwargs)
    with pytest.raises(ValueError):
        AnalysisSummary('{"value":NaN}')


def test_summary_content_ids_match_records_and_depend_on_product_of():
    result = harmonic_thermodynamics([2], [0, 300])
    edges = (RunEdge('source_0', 'files', 'f' * 8),)

    def ids(fields):  # JSON keys are sorted, so read in binding order
        return [fields[b.field]['content_id'] for b in bound_values(result)]

    plain = _summary(result).value['fields']
    linked = _summary(result, product_of=edges).value['fields']
    assert ids(linked) == [content_id(r) for r in records(result, product_of=edges)]
    assert ids(plain) == [content_id(r) for r in records(result)]
    assert ids(plain) != ids(linked)
    bare = _summary(result, product_of=edges, field_values=False).value['fields']
    assert all('value' not in f for f in bare.values())
    assert ids(bare) == ids(linked)


def test_summary_rejects_duplicate_binding_names():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Fake:
        def _bound_values(self):
            one = defs.BoundValue("same", defs.FieldBinding(defs.FERMI_ENERGY), 1.0)
            return (one, one)

    with pytest.raises(ValueError, match="duplicate"):
        analysis_summary(Fake(), algorithm="x", parameters={}, selection={}, sources=[], assumptions=[])


def test_summary_energy_errors_has_raw_and_corrected_fields():
    from httk.analyse.matsci.mlip import energy_errors

    result = energy_errors([0.0, 0.0], [0.0, 10.0], atom_counts=[1, 4], offset_per_atom=1.0)
    fields = _summary(result).value["fields"]
    raw, corrected = fields["energy_prediction_errors"], fields["corrected_energy_prediction_errors"]
    assert raw["definition"] == corrected["definition"] == defs.ENERGY_PREDICTION_ERRORS
    assert raw["content_id"] != corrected["content_id"]
    assert raw["value"]["offset_per_atom"] is None and corrected["value"]["offset_per_atom"] == 1.0
