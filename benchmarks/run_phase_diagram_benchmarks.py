#!/usr/bin/env python3
"""Run resumable httk/ASE comparisons, with one guarded process per timing.

Linux, GNU timeout, and the benchmark extra are required. Numerical libraries
are imported only in workers, after memory and time limits have been installed.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from _phase_diagram_worker import validate_case

WORKER = Path(__file__).with_name('_phase_diagram_worker.py')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')
CHECK_TOLERANCE = 1e-7


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-counts', type=_positive, nargs='+', default=[20, 40, 80])
    parser.add_argument('--phase-sweep-species', type=_positive, default=3)
    parser.add_argument('--species-counts', type=_positive, nargs='+', default=[2, 3, 4, 6, 8, 10, 12])
    parser.add_argument('--species-sweep-phases', type=_positive, default=30)
    parser.add_argument('--atom-count', type=_positive, default=10_000)
    parser.add_argument('--seed', type=int, default=20_260_731)
    parser.add_argument('--repeats', type=_positive, default=3, help='fresh timed workers; report median')
    parser.add_argument('--warmups', type=int, default=0, help='untimed runs within each fresh worker (default: 0)')
    parser.add_argument('--phase-lines', action='store_true', help='also time first lazy httk phase-line access')
    parser.add_argument(
        '--httk-solver',
        choices=('simplex', 'highs'),
        default='simplex',
        help='HTTK solver (highs requires the highs extra; one solver thread)',
    )
    parser.add_argument('--max-rss-gb', type=_positive_float, default=1.0, help='sampled group RSS limit in GiB')
    parser.add_argument(
        '--as-gb', type=_positive_float, default=2.0, help='hard per-process address-space limit in GiB'
    )
    parser.add_argument('--timeout', type=_positive_float, default=60.0, help='seconds per worker, INCLUDING setup')
    parser.add_argument('--threads', type=_positive, default=1, help='OMP/OpenBLAS/MKL/BLIS threads per worker')
    parser.add_argument('--json', type=Path, default=Path('benchmark-results/phase-diagram.json'), dest='json_path')
    parser.add_argument('--report', type=Path, help='Markdown output (default: JSON path with .md suffix)')
    parser.add_argument('--resume', action='store_true', help='run only missing measurements in a matching checkpoint')
    parser.add_argument('--report-only', action='store_true', help='render saved JSON without running measurements')
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error('--warmups must be non-negative')
    args.report = args.report or args.json_path.with_suffix('.md')
    if args.report.resolve() == args.json_path.resolve():
        parser.error('JSON and report paths must differ')
    return args


def _cases(args):
    cases = []
    for sweep, values in (
        ('phases', [(args.phase_sweep_species, n) for n in args.phase_counts]),
        ('species', [(s, args.species_sweep_phases) for s in args.species_counts]),
    ):
        for species, phases in values:
            case = {
                'sweep': sweep,
                'species': species,
                'phases': phases,
                'seed': args.seed + 1_000_003 * species + phases,
            }
            validate_case(case, args.atom_count)
            if case not in cases:
                cases.append(case)
    return cases


def _source_state():
    """Fingerprint imported source locations without importing numerical packages."""
    state = {}
    for name in ('httk.core', 'httk.atomistic', 'httk.analyse'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f'{name} must be installed')
        directory = Path(spec.origin).parent.resolve()
        digest = hashlib.sha256()
        for path in sorted(directory.rglob('*.py')):
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
        head = subprocess.run(
            ['git', '-C', str(directory), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True,
            text=True,
            check=False,
        )
        state[name] = {
            'path': str(directory),
            'sha256': digest.hexdigest(),
            'git_head': head.stdout.strip() or None,
            'tracked_dirty': bool(dirty.stdout),
        }
    state['benchmark_scripts'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), WORKER)
    }
    return state


def _metadata(args):
    packages = {}
    for name in ('httk-core', 'httk-atomistic', 'httk-analyse', 'numpy', 'ase', 'scipy', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"{name} missing; install this checkout with pip install -e '.[benchmark]'") from exc
    if args.httk_solver == 'highs':
        try:
            packages['highspy'] = importlib.metadata.version('highspy')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "highspy missing; install this checkout with pip install -e '.[benchmark,highs]'"
            ) from exc
    cpu = next(
        (
            line.split(':', 1)[1].strip()
            for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name')
        ),
        platform.machine(),
    )
    return {
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'cpu': cpu,
        'packages': packages,
        'sources': _source_state(),
        'parameters': {
            key: value
            for key, value in vars(args).items()
            if key not in ('json_path', 'report', 'resume', 'report_only')
        },
        'thread_environment': {name: str(args.threads) for name in THREAD_VARIABLES},
        'httk_tolerance': 1e-8,
        'comparison_absolute_tolerance': CHECK_TOLERANCE,
        'ase_tolerance': 'ASE defaults (Qhull construction; decompose eps=1e-14)',
    }


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Importing core's atomic helper initializes its optional NumPy views.
    # Keep this supervisor stdlib-only; all scientific imports belong in children.
    with TemporaryDirectory(prefix='.phase-benchmark-', dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(text, encoding='utf-8')
        staged.replace(path)


def _checkpoint(path, payload):
    _write(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _run_measurement(request, args):
    """Place GNU timeout inside memguard's process group so descendants are killed."""
    with TemporaryDirectory(prefix='httk-phase-benchmark-') as temporary:
        directory = Path(temporary)
        input_path, output_path = directory / 'request.json', directory / 'response.json'
        input_path.write_text(json.dumps(request), encoding='utf-8')
        command = [
            sys.executable,
            '-m',
            'httk.core.memguard',
            '--max-rss-gb',
            str(args.max_rss_gb),
            '--as-gb',
            str(args.as_gb),
            '--interval',
            '0.1',
            '--',
            'timeout',
            '--verbose',
            '--signal=TERM',
            '--kill-after=3s',
            f'{args.timeout}s',
            sys.executable,
            str(WORKER),
            str(input_path),
            str(output_path),
        ]
        environment = dict(os.environ, **{name: str(args.threads) for name in THREAD_VARIABLES}, LC_ALL='C')
        # A file avoids buffering arbitrary native-library diagnostics in supervisor RAM.
        with (directory / 'worker.log').open('w+') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, start_new_session=True)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                # memguard forwards TERM; GNU timeout kills unresponsive workers after its grace period.
                process.send_signal(signal.SIGTERM)
                process.wait()
                raise
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 16_384))
            diagnostic = log.read()
        result = {'status': 'error', 'seconds': None, 'peak_rss_bytes': None, 'summary': {}, 'error': None}
        if output_path.exists():
            result.update(json.loads(output_path.read_text(encoding='utf-8')))
        if returncode != 0:
            if 'budget - killing the group' in diagnostic:
                result['status'] = 'memory_limit'
            elif returncode == 124 or 'timeout: sending signal' in diagnostic:
                result['status'] = 'timeout'
            elif result['status'] != 'memory_limit':
                result['status'] = 'error'
            result['error'] = result['error'] or f'worker exited with status {returncode}'
            result['seconds'] = None
        elif not output_path.exists():
            result['error'] = 'worker exited without a result'
        result.update(returncode=returncode, diagnostic=diagnostic)
        return result


def _agreement(records):
    """Compare every successful repetition, without requiring unique decompositions."""
    good = [r for r in records if r['status'] == 'ok']
    summaries = [r['summary'] for r in good if r['metric'] in ('httk', 'ase_full')]
    checks = {}
    for key in ('max_composition_residual', 'max_energy_residual', 'max_weight_sum_error'):
        values = [s[key] for s in summaries if key in s]
        checks[key] = max(values) if values else None
    weights = [s['min_weight'] for s in summaries if 'min_weight' in s]
    checks['min_weight'] = min(weights) if weights else None
    httk = [r['summary'] for r in good if r['metric'] == 'httk']
    ase = [r['summary'] for r in good if r['metric'] in ('ase_construct', 'ase_full')]
    checks['stable_membership_matches'] = (
        all(h['stable_indices'] == a['stable_indices'] for h in httk for a in ase) if httk and ase else None
    )
    energies = [a['hull_energies'] for a in ase if 'hull_energies' in a]
    checks['max_hull_energy_difference'] = (
        max(abs(x - y) for h in httk for a in energies for x, y in zip(h['hull_energies'], a, strict=True))
        if httk and energies
        else None
    )
    checked = checks['max_hull_energy_difference'] is not None
    checks['numerical_checks_pass'] = (
        all(
            checks[key] is not None and checks[key] <= CHECK_TOLERANCE
            for key in (
                'max_composition_residual',
                'max_energy_residual',
                'max_weight_sum_error',
                'max_hull_energy_difference',
            )
        )
        and checks['min_weight'] is not None
        and checks['min_weight'] >= -CHECK_TOLERANCE
        if checked
        else None
    )
    return checks


def _report(payload):
    metadata = payload['metadata']
    lines = [
        '# Phase-diagram benchmark',
        '',
        f"Started: {payload['started_at']}",
        '',
        'Times are medians of successful fresh-worker trials, in milliseconds. RSS is the maximum',
        'successful worker peak in MiB, INCLUDING imports, input generation, warmups and validation.',
        'Timeout includes that setup; phase-line timing excludes construction but its RSS does not.',
        f"HTTK solver: {metadata.get('parameters', {}).get('httk_solver', 'simplex')}.",
        'The HTTK HiGHS solver uses one solver thread; --threads controls NumPy/BLAS environment variables only.',
        'Failures are not timings. Partial medians are labelled by the completed/expected count.',
        '',
        'httk includes hull membership, energy above hull and unstable decompositions. `ase_construct`',
        'builds the hull only; `ase_full` also queries every unstable input. `httk_phase_lines` is',
        'optional first access. LP counts are estimates, not instrumentation.',
        '',
        '| Sweep | S | N | Metric | OK / expected | Median ms | Peak MiB | Outcomes |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |',
    ]
    for index, case in enumerate(payload['cases']):
        records = [r for r in payload['results'] if r['case_index'] == index]
        for metric in payload['metrics']:
            trials = [r for r in records if r['metric'] == metric]
            good = [r for r in trials if r['status'] == 'ok']
            timing = f"{1000 * statistics.median(r['seconds'] for r in good):.3f}" if good else '—'
            rss = f"{max(r['peak_rss_bytes'] for r in good) / 2**20:.2f}" if good else '—'
            outcomes = ', '.join(sorted({r['status'] for r in trials})) or 'pending'
            lines.append(
                f"| {case['sweep']} | {case['species']} | {case['phases']} | {metric} | "
                f"{len(good)} / {metadata['parameters']['repeats']} | {timing} | {rss} | {outcomes} |"
            )
    lines += [
        '',
        '## Correctness and output sizes',
        '',
        'Absolute check threshold: 1e-7 (energy in eV/atom; compositions and weights dimensionless).',
        'Membership is reported separately: coplanar hull membership conventions can differ.',
        'Different valid decompositions are accepted. Missing comparisons are not passes.',
        '',
    ]
    for index, case in enumerate(payload['cases']):
        records = [r for r in payload['results'] if r['case_index'] == index]
        checks = _agreement(records)
        lines += [
            f"### {case['sweep']}: S={case['species']}, N={case['phases']}",
            '',
            '```json',
            json.dumps(checks, indent=2),
            '```',
            '',
        ]
        medians = {
            metric: statistics.median(r['seconds'] for r in records if r['metric'] == metric and r['status'] == 'ok')
            for metric in ('httk', 'ase_full')
            if any(r['metric'] == metric and r['status'] == 'ok' for r in records)
        }
        if len(medians) == 2 and medians['ase_full'] > 0:
            lines += [
                f"httk / ASE full time: {medians['httk'] / medians['ase_full']:.2f}× (available successful trials).",
                '',
            ]
        for metric in payload['metrics']:
            trial = next((r for r in records if r['metric'] == metric and r['status'] == 'ok'), None)
            if trial:
                summary = trial['summary']
                sizes = {
                    key: summary[key] for key in ('lower_facets', 'phase_lines', 'lp_solves_estimate') if key in summary
                }
                sizes['stable_phases'] = len(summary['stable_indices'])
                lines.append(f'- {metric}: {sizes}')
        lines += ['']
    failures = [r for r in payload['results'] if r['status'] != 'ok']
    if failures:
        lines += ['## Failed measurements', '']
        for result in failures:
            lines += [
                f"Case {result['case_index']}, {result['metric']}, trial {result['repeat'] + 1}: {result['status']}",
                '',
                '```text',
                result['error'] or '',
                result['diagnostic'],
                '```',
                '',
            ]
    lines += [
        '## Reproducibility',
        '',
        'Synthetic unique integer compositions, pure-element endpoints and random energies;',
        'these are not representative real-material data. Use multiple seeds before generalizing.',
        'No other concurrent workload is controlled. Python-source hashes detect live edits;',
        'use frozen environments/checkouts for publication-quality runs. Non-Python data and',
        'same-version dependency binary replacements are not fingerprinted.',
        '',
        '```json',
        json.dumps(metadata, indent=2, sort_keys=True),
        '```',
        '',
    ]
    return '\n'.join(lines)


def _execute(args):
    if args.report_only:
        payload = json.loads(args.json_path.read_text(encoding='utf-8'))
        _write(args.report, _report(payload))
        print(f'Wrote {args.report}')
        return 0
    if sys.platform != 'linux' or shutil.which('timeout') is None:
        raise RuntimeError('guarded benchmarks require Linux /proc and GNU timeout; no unguarded fallback')
    cases = _cases(args)
    metadata = _metadata(args)
    metrics = ['httk', 'ase_construct', 'ase_full'] + (['httk_phase_lines'] if args.phase_lines else [])
    if args.resume:
        payload = json.loads(args.json_path.read_text(encoding='utf-8'))
        if payload.get('format') != 1 or payload['metadata'] != metadata or payload['cases'] != cases:
            raise ValueError(
                'checkpoint does not match parameters/software; use the original environment or a new --json'
            )
    else:
        if args.json_path.exists():
            raise ValueError(f'{args.json_path} exists; use --resume or a new --json path')
        if args.report.exists():
            raise ValueError(f'{args.report} exists; choose a new --report path')
        payload = {
            'format': 1,
            'started_at': datetime.now(UTC).isoformat(),
            'metadata': metadata,
            'cases': cases,
            'metrics': metrics,
            'results': [],
        }
        _checkpoint(args.json_path, payload)
    try:
        for index, case in enumerate(cases):
            for metric in metrics:
                for repeat in range(args.repeats):
                    previous = [r for r in payload['results'] if r['case_index'] == index and r['metric'] == metric]
                    # One failed trial is enough for this case/mode; do not repeatedly exhaust its budget.
                    if any(r['status'] != 'ok' for r in previous):
                        break
                    if any(r['repeat'] == repeat for r in previous):
                        continue
                    if _source_state() != metadata['sources']:
                        raise RuntimeError('source changed during the run; stopping to avoid mixed-software results')
                    print(
                        f"{case['sweep']} S={case['species']} N={case['phases']} {metric} {repeat + 1}/{args.repeats}",
                        flush=True,
                    )
                    request = {
                        'case': case,
                        'metric': metric,
                        'atom_count': args.atom_count,
                        'warmups': args.warmups,
                        'solver': args.httk_solver,
                    }
                    result = _run_measurement(request, args)
                    changed = _source_state() != metadata['sources']
                    if changed:
                        result.update(status='invalidated', seconds=None, error='source changed while worker ran')
                    result.update(case_index=index, metric=metric, repeat=repeat)
                    payload['results'].append(result)
                    _checkpoint(args.json_path, payload)
                    print(f"  {result['status']}", flush=True)
                    if changed:
                        raise RuntimeError(result['error'])
    finally:
        _write(args.report, _report(payload))
    print(f'Checkpoint: {args.json_path}\nReport: {args.report}')
    failed = any(r['status'] != 'ok' for r in payload['results'])
    mismatch = any(
        _agreement([r for r in payload['results'] if r['case_index'] == i])['numerical_checks_pass'] is False
        for i in range(len(cases))
    )
    return 1 if failed or mismatch else 0


def main(argv=None) -> int:
    """Run or resume guarded measurements, or render a saved report."""
    args = _parse_args(argv)
    try:
        if args.report_only:
            return _execute(args)
        if sys.platform != 'linux':
            raise RuntimeError('guarded benchmarks require Linux /proc and GNU timeout; no unguarded fallback')
        # Serialize checkpoint writers despite atomic replacement changing the JSON inode.
        import fcntl

        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        with args.json_path.with_suffix(args.json_path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _execute(args)
    except KeyboardInterrupt:
        print(
            'Interrupted; completed measurements are checkpointed. Resume with the same arguments and --resume.',
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Benchmark: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
