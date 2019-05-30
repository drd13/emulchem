#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `emulchem` package."""


import unittest

from emulchem import emulchem


class TestEmulchem(unittest.TestCase):
    """Tests for `emulchem` package."""

    def setUp(self):
        self.CO_emulator = ChemistryEmulator("CO")
        """Set up test fixtures, if any."""

    def tearDown(self):
        self.CO_emulator.dispose()
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""
