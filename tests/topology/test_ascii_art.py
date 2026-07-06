"""Tests for ascii_art_heterocycle: ASCII art → AtomicSystem parsing."""
import numpy as np
import pytest
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES


class TestParseAsciiArt:
    def test_all_examples_parse(self):
        """All ASCII examples should parse without error."""
        for name, art in ASCII_EXAMPLES.items():
            atoms = parse_ascii_art(art)
            assert atoms.natoms > 0, f"{name}: no atoms"

    def test_naphthalene_natoms(self):
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        assert atoms.natoms == 10

    def test_bonds_not_empty(self):
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        assert atoms.bonds is not None and len(atoms.bonds) > 0

    def test_enames_original_set(self):
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        assert getattr(atoms, '_enames_original', None) is not None

    def test_coordinates_finite(self):
        for name, art in ASCII_EXAMPLES.items():
            atoms = parse_ascii_art(art)
            assert np.all(np.isfinite(atoms.apos)), f"{name}: non-finite coordinates"
