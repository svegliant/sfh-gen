"""
Unit tests for the parts of the pipeline that need no OpenMolcas.

These cover the logic whose failure is *silent*: a wrong CSF index, a
rank-deficient CI matrix, a degenerate active pair or a mis-parsed RasOrb all
produce a complete run and a plausible-looking 5x5 matrix. Everything here
runs in well under a second and needs only numpy, h5py and pyyaml.

    pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfh_gen.build_SF_H import (          # noqa: E402
    CSF_LABELS, GROUND_CSF, SF_STATES, csf_index, build_hamiltonian,
    extract_sf_states, format_hamiltonian, matrix_lines, save_hamiltonian,
    SFHamiltonianError, AU2EV,
)
from sfh_gen.instrumentation import (     # noqa: E402
    format_duration, TimingSummary,
)
from sfh_gen.cli import default_config_text   # noqa: E402
from sfh_gen.config_parser import (       # noqa: E402
    n_csf, ConfigError, load_config, split_nprocs, _SCHEMA,
)
from sfh_gen.split_dimer import (           # noqa: E402
    check_fragments_contiguous, SplitDimerError,
)
from sfh_gen.local_orbitals import (      # noqa: E402
    vec_lowdin, check_active_occupations, check_orbital_types,
    find_atom_offset, LocalOrbitalsError,
)


# ── The CSF table ─────────────────────────────────────────────────────────
# The five diabatic states are picked out of the CI vectors purely by index.
# If this table drifts, every coupling is silently attributed to the wrong
# pair of states.

def test_csf_table_is_the_cas44_singlet_space():
    assert len(CSF_LABELS) == 20
    assert len(set(CSF_LABELS)) == 20, "duplicate CSF label"
    assert n_csf(4, 4, 1) == len(CSF_LABELS)


def test_ground_csf_index_is_stable():
    assert csf_index(GROUND_CSF) == 5
    assert GROUND_CSF == "2020"


def test_sf_state_indices_are_stable():
    assert [csf_index(csf) for _, csf in SF_STATES] == [8, 6, 15, 1, 14]


def test_csf_index_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="not a CAS"):
        csf_index("2222")


# ── H = C^T diag(E) C ─────────────────────────────────────────────────────

def _orthogonal(n, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


def test_hamiltonian_reproduces_its_own_eigenvalues():
    """The transform must be a similarity transform, not any old congruence."""
    n = len(CSF_LABELS)
    C = _orthogonal(n)
    E = np.linspace(-1.0, -0.5, n)
    H = build_hamiltonian(C, E)
    assert np.allclose(np.sort(np.linalg.eigvalsh(H)), np.sort(E))
    assert np.allclose(H, H.T), "the Hamiltonian must be symmetric"


def test_fewer_roots_than_csfs_is_refused():
    """A rank-deficient projection looks plausible and means nothing."""
    C = _orthogonal(len(CSF_LABELS))[:5]
    with pytest.raises(SFHamiltonianError, match="roots for 20 CSFs"):
        build_hamiltonian(C, np.linspace(-1, -0.5, 5))


def test_non_orthonormal_ci_vectors_are_refused():
    n = len(CSF_LABELS)
    C = _orthogonal(n)
    C[0, 0] += 0.1                      # break orthonormality
    with pytest.raises(SFHamiltonianError, match="not orthonormal"):
        build_hamiltonian(C, np.linspace(-1, -0.5, n))


def test_extract_sf_states_shifts_the_diagonal_and_converts_to_ev():
    n = len(CSF_LABELS)
    H = np.diag(np.linspace(-1.0, -0.5, n))
    csf_labels = [csf for _, csf in SF_STATES]
    zero = H[csf_index(GROUND_CSF), csf_index(GROUND_CSF)]
    block = extract_sf_states(H, csf_labels, zero)
    assert block.shape == (5, 5)
    expected = (H[csf_index(csf_labels[0]), csf_index(csf_labels[0])] - zero) * AU2EV
    assert block[0, 0] == pytest.approx(expected)


# ── The written Hamiltonian ───────────────────────────────────────────────

def test_saved_hamiltonian_round_trips_and_carries_provenance(tmp_path):
    H = np.arange(25, dtype=float).reshape(5, 5)
    H = H + H.T
    out = tmp_path / "H_SF.dat"
    save_hamiltonian(H, [n for n, _ in SF_STATES], str(out), source="x.h5")

    text = out.read_text()
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    # Matched per line: the header embeds an absolute source path, so a bare
    # substring test over the whole file can be satisfied by the path instead.
    assert any(ln.startswith("# units") and "eV" in ln for ln in header)
    assert any(ln.startswith("# energy zero") and f"eps(|{GROUND_CSF}>)" in ln
               for ln in header)
    assert any(ln.startswith("# produced by") and "sfh-gen" in ln
               for ln in header)

    back = np.loadtxt(out, comments="#", usecols=range(5))
    assert np.allclose(back, H)
    assert not list(tmp_path.glob("*.part")), "temporary file left behind"


# ── What the run prints ───────────────────────────────────────────────────
# The result block repeats the matrix that goes into H_SF.dat, so that one job
# log holds both. Two formattings of the same numbers would drift apart
# unnoticed; one function produces the rows for both.

def test_the_printed_matrix_is_the_one_written_out(tmp_path):
    H = np.arange(25, dtype=float).reshape(5, 5)
    H = H + H.T
    names = [n for n, _ in SF_STATES]
    csfs = [c for _, c in SF_STATES]

    out = tmp_path / "H_SF.dat"
    save_hamiltonian(H, names, str(out))
    written = out.read_text()
    printed = format_hamiltonian(H, names, csfs)

    for row in matrix_lines(H, names)[1:]:
        assert f"{row}\n" in written
        assert row in printed
    for name, csf in SF_STATES:
        assert f"{name} = {csf}" in printed


@pytest.mark.parametrize("seconds,expected", [
    (0.44, "0.4 s"),
    (59.94, "59.9 s"),
    (59.96, "1 m 00.0 s"),     # rounding must not produce "60.0 s"
    (74.3, "1 m 14.3 s"),
    (4327.6, "1 h 12 m 07 s"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_the_timing_table_lists_every_step_and_the_wall_clock():
    """Concurrent steps are each wall clock, so the rows may add up to more
    than the run took; the total is measured separately, not summed."""
    timing = TimingSummary()
    timing.record("01  CASSCF on monomer 1", 30.0)
    timing.record("02  CASSCF on monomer 2", 28.0)
    text = timing.format_summary(wall=31.0, concurrent=True)
    assert "01  CASSCF on monomer 1" in text
    assert "02  CASSCF on monomer 2" in text
    assert "total wall clock" in text and "31.0 s" in text
    assert "concurrently" in text


# ── Weyl's formula ────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_el,n_orb,spin,expected", [
    (4, 4, 1, 20),      # the space this pipeline is built for
    (2, 2, 1, 3),
    (6, 6, 1, 175),
])
def test_n_csf(n_el, n_orb, spin, expected):
    assert n_csf(n_el, n_orb, spin) == expected


def test_n_csf_rejects_an_impossible_space():
    with pytest.raises(ConfigError):
        n_csf(3, 4, 1)          # odd electron count in a singlet


# ── Splitting the processes between the concurrent steps ──────────────────

@pytest.mark.parametrize("total,expected", [
    (None, (None, None)), (1, (1, 1)), (2, (1, 1)),
    (3, (2, 1)), (12, (6, 6)), (13, (7, 6)),
])
def test_split_nprocs_two_ways(total, expected):
    assert split_nprocs(total, 2) == expected


@pytest.mark.parametrize("total,expected", [
    (None, (None, None, None)), (1, (1, 1, 1)), (3, (1, 1, 1)),
    (7, (3, 2, 2)), (12, (4, 4, 4)), (13, (5, 4, 4)),
])
def test_split_nprocs_three_ways(total, expected):
    """01, 02 and 03 overlap by default, so the allocation divides by three."""
    assert split_nprocs(total, 3) == expected


@pytest.mark.parametrize("total", [1, 2, 3, 7, 12, 13, 24, 97])
@pytest.mark.parametrize("n_ways", [1, 2, 3, 4])
def test_split_nprocs_conserves_the_allocation(total, n_ways):
    shares = split_nprocs(total, n_ways)
    assert len(shares) == n_ways
    assert all(n >= 1 for n in shares), "a run cannot be given zero processes"
    if total >= n_ways:
        assert sum(shares) == total, "processes must be neither lost nor invented"
        assert max(shares) - min(shares) <= 1, "the split must be even"


# ── Loewdin orthogonalization ─────────────────────────────────────────────

def test_vec_lowdin_produces_an_orthonormal_set():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(12, 12))
    S = A @ A.T + 12 * np.eye(12)          # a plausible overlap matrix
    B = rng.normal(size=(12, 4))
    C = vec_lowdin(B, S)
    assert np.allclose(C.T @ S @ C, np.eye(4), atol=1e-10)


def test_vec_lowdin_is_the_minimal_change():
    """Symmetric orthogonalization is the orthonormal set closest to the input.

    This is its defining property and the reason it is used here: Gram-Schmidt
    also orthonormalizes, but it leaves the first column untouched and pushes
    all the error into the last, which would bias the diabatic basis towards
    whichever monomer happens to be written first in the file.
    """
    rng = np.random.default_rng(2)
    S = np.eye(6)
    B = rng.normal(size=(6, 3))
    B /= np.linalg.norm(B, axis=0)

    lowdin = vec_lowdin(B, S)
    gram_schmidt, _ = np.linalg.qr(B)

    assert np.linalg.norm(lowdin - B) < np.linalg.norm(gram_schmidt - B)


def test_vec_lowdin_barely_moves_a_nearly_orthogonal_set():
    """The realistic case: monomer orbitals on well-separated fragments overlap
    by a few percent, and the orthogonalization must perturb them only slightly
    or it would destroy the locality that defines the diabatic states."""
    rng = np.random.default_rng(4)
    B = np.eye(8)[:, :4] + 0.02 * rng.normal(size=(8, 4))
    B /= np.linalg.norm(B, axis=0)
    C = vec_lowdin(B, np.eye(8))
    overlaps = np.abs(np.sum(B * C, axis=0))
    assert (overlaps > 0.99).all(), f"Loewdin moved the orbitals too far: {overlaps}"


def test_vec_lowdin_does_not_depend_on_column_order():
    rng = np.random.default_rng(3)
    S = np.eye(6)
    B = rng.normal(size=(6, 3))
    order = [2, 0, 1]
    C = vec_lowdin(B, S)
    C_perm = vec_lowdin(B[:, order], S)
    assert np.allclose(C[:, order], C_perm, atol=1e-12)


def test_the_ground_weight_threshold_has_exactly_one_value():
    """local_orbitals keeps a fallback copy for standalone use; if the two ever
    disagree the banner would quote a threshold the check does not apply."""
    import sfh_gen.local_orbitals as lo
    import sfh_gen.build_SF_H as bh
    assert lo.GROUND_WEIGHT_MIN == bh.GROUND_WEIGHT_MIN


# ── The guards on the monomer orbitals ────────────────────────────────────

def test_degenerate_active_occupations_are_fatal():
    """A state-averaged RasOrb gives exactly 1.0/1.0 and an arbitrary rotation."""
    with pytest.raises(LocalOrbitalsError, match="degenerate"):
        check_active_occupations([2.0, 1.0, 1.0, 0.0], ["I", "2", "2", "S"], "mono1")


def test_lumo_first_ordering_is_recovered_not_rejected():
    act, swapped = check_active_occupations(
        [2.0, 0.0465, 1.9535, 0.0], ["I", "2", "2", "S"], "mono1")
    assert swapped is True
    assert act == [2, 1], "the HOMO must come first after the swap"


def test_descending_occupations_are_left_alone():
    act, swapped = check_active_occupations(
        [2.0, 1.9535, 0.0465, 0.0], ["I", "2", "2", "S"], "mono1")
    assert swapped is False
    assert act == [1, 2]


def test_unhandled_orbital_types_are_refused():
    with pytest.raises(LocalOrbitalsError, match="orbital type"):
        check_orbital_types(["I", "F", "2", "2", "S"], "mono1")


# ── Locating a monomer inside the dimer ───────────────────────────────────

def test_find_atom_offset_locates_a_contiguous_block():
    dimer = np.arange(30, dtype=float).reshape(10, 3)
    assert find_atom_offset(dimer[4:9], dimer, "mono2") == 4


def test_find_atom_offset_is_absolute_not_relative():
    """rtol=0: a 1e-4 error on a large coordinate must not pass a 1e-6 test."""
    dimer = np.full((6, 3), 100.0) + np.arange(18).reshape(6, 3)
    mono = dimer[2:5].copy()
    mono[0, 0] += 1e-4
    with pytest.raises(LocalOrbitalsError, match="could not locate"):
        find_atom_offset(mono, dimer, "mono1", tol=1e-6)


# ── Splitting the dimer ───────────────────────────────────────────────────
# The embedding maps basis functions by index, so the monomers must be
# consecutive blocks starting at atom 0. These run without MDAnalysis.

def test_two_contiguous_fragments_are_accepted():
    check_fragments_contiguous([np.arange(0, 30), np.arange(30, 60)])


def test_interleaved_fragments_are_refused():
    """The case the check exists for: alternating atoms cannot be embedded."""
    with pytest.raises(SplitDimerError, match="contiguous block"):
        check_fragments_contiguous([np.arange(0, 60, 2), np.arange(1, 60, 2)])


def test_fragments_not_starting_at_zero_are_refused():
    with pytest.raises(SplitDimerError, match="expected to start at 0"):
        check_fragments_contiguous([np.arange(5, 35), np.arange(35, 65)])


def test_a_fragment_with_a_hole_is_refused():
    with_hole = np.concatenate([np.arange(0, 10), np.arange(11, 31)])
    with pytest.raises(SplitDimerError, match="contiguous block"):
        check_fragments_contiguous([with_hole, np.arange(31, 61)])


def test_every_module_has_no_undefined_names():
    """A NameError inside a rarely-taken branch ships silently otherwise.

    The automatic dimer split is the branch at risk: every test and example
    passes the monomer geometries explicitly, so nothing else exercises it.
    """
    api = pytest.importorskip("pyflakes.api")
    reporter = pytest.importorskip("pyflakes.reporter")
    import io as _io
    pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "sfh_gen")
    out, err = _io.StringIO(), _io.StringIO()
    rep = reporter.Reporter(out, err)
    for name in sorted(os.listdir(pkg)):
        if name.endswith(".py"):
            api.checkPath(os.path.join(pkg, name), rep)
    findings = [ln for ln in out.getvalue().splitlines() if "undefined name" in ln]
    assert not findings, "undefined names:\n" + "\n".join(findings)


# ── The shipped default config ────────────────────────────────────────────
# `sfh-gen` writes default_config.yaml into an empty directory, and
# config_parser._SCHEMA decides what is accepted. Nothing links the two, so
# they can drift: a key added to one and forgotten in the other gives a user a
# generated config that fails its own validation.

def test_the_generated_config_is_valid(tmp_path):
    """What sfh-gen writes must be something sfh-gen accepts."""
    (tmp_path / "dimer.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(default_config_text())
    cfg = load_config(str(cfg_path))         # raises ConfigError if not
    assert cfg.system.name == "dimer"
    assert cfg.dimer.n_roots == 20


def test_the_generated_config_uses_no_key_the_schema_rejects(tmp_path):
    import yaml
    body = yaml.safe_load(default_config_text())
    assert set(body) <= set(_SCHEMA), f"unknown sections: {set(body) - set(_SCHEMA)}"
    for section, keys in body.items():
        extra = set(keys or {}) - _SCHEMA[section]
        assert not extra, f"[{section}] generated but rejected: {sorted(extra)}"


def test_the_generated_config_mentions_every_accepted_key():
    """Every key a user may set should appear in the file they are handed,
    set or commented out, or they will never learn it exists."""
    text = default_config_text()
    missing = [f"{section}.{key}"
               for section, keys in _SCHEMA.items() for key in keys
               if key not in text]
    assert not missing, f"accepted but undocumented in the template: {missing}"


# ── Config validation ─────────────────────────────────────────────────────

def _write_config(tmp_path, body):
    (tmp_path / "dimer.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return str(p)


BASE = """
paths:
  geometry: dimer.xyz
  openmolcas: pymolcas
system:
  name: t
basis:
  name: 6-31G
"""


def test_a_minimal_config_loads(tmp_path):
    cfg = load_config(_write_config(tmp_path, BASE))
    assert cfg.dimer.n_roots == 20, "n_roots must default to the CSF count"
    assert os.path.isabs(cfg.paths.geometry)


def test_load_config_does_not_change_the_working_directory(tmp_path):
    before = os.getcwd()
    load_config(_write_config(tmp_path, BASE))
    assert os.getcwd() == before


def test_a_misspelled_key_is_refused(tmp_path):
    body = BASE + "monomer:\n  natural_orbital_rooot: 1\n"
    with pytest.raises(ConfigError, match="natural_orbital_root"):
        load_config(_write_config(tmp_path, body))


def test_an_unknown_section_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="Unknown configuration section"):
        load_config(_write_config(tmp_path, BASE + "monomers:\n  n_roots: 3\n"))


def test_a_non_singlet_is_refused_at_load_time(tmp_path):
    body = BASE.replace("  name: t", "  name: t\n  spin: 3")
    with pytest.raises(ConfigError, match="spin must be 1"):
        load_config(_write_config(tmp_path, body))


def test_a_non_numeric_value_is_a_config_error_not_a_traceback(tmp_path):
    body = BASE + "integrals:\n  cutoff: abc\n"
    with pytest.raises(ConfigError, match="must be float"):
        load_config(_write_config(tmp_path, body))


def test_malformed_yaml_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(_write_config(tmp_path, 'paths:\n  geometry: "unclosed\n'))


def test_wrong_dimer_active_space_is_refused(tmp_path):
    body = BASE + "dimer:\n  n_electrons: 6\n  n_orbitals: 6\n"
    with pytest.raises(ConfigError, match="must be .4,4."):
        load_config(_write_config(tmp_path, body))


def test_n_roots_must_equal_the_csf_count(tmp_path):
    with pytest.raises(ConfigError, match="20 CSFs"):
        load_config(_write_config(tmp_path, BASE + "dimer:\n  n_roots: 5\n"))


def test_basis_name_and_per_element_are_exclusive(tmp_path):
    body = BASE + "  per_element:\n    C: 6-31G*\n"
    with pytest.raises(ConfigError, match="not both"):
        load_config(_write_config(tmp_path, body))
