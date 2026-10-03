"""Provenance envelopes retain file identities and immutable numerical results."""

import hashlib
import json

import pytest

from httk.analyse.summary import AnalysisSummary, analysis_summary
from httk.analyse.matsci.phonons import harmonic_thermodynamics


def test_summary_roundtrip_hash_selection_and_immutability(tmp_path):
    source = tmp_path / 'modes.csv'
    source.write_text('frequency_THz\n2\n')
    result = harmonic_thermodynamics([2], [0, 300])
    summary = analysis_summary(
        result,
        algorithm='harmonic_thermodynamics',
        units={'free_energy': 'eV'},
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


def test_summary_rejects_nonfinite_and_encodes_complex(tmp_path):
    kwargs = dict(
        algorithm='scattering', units={'values': '1'}, parameters={}, selection={}, sources=[], assumptions=[]
    )
    result = analysis_summary({'values': [1 + 2j]}, **kwargs)
    assert result.value['result']['values'] == [{'real': 1, 'imaginary': 2}]
    with pytest.raises(ValueError):
        analysis_summary({'values': [float('nan')]}, **kwargs)
    with pytest.raises(ValueError):
        AnalysisSummary('{"value":NaN}')
