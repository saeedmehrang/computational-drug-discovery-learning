"""
Pytest test suite for torsion extraction in torsional diffusion model.
Tests that only heavy-atom torsions are counted correctly.

Run with: pytest test_torsion_fix.py -v
Run with output: pytest test_torsion_fix.py -v -s
"""

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torsional_diffusion import MolecularTorsionAnalyzer


@pytest.fixture
def analyzer():
    """Fixture providing a MolecularTorsionAnalyzer instance."""
    return MolecularTorsionAnalyzer()


@pytest.fixture
def molecule_factory():
    """
    Fixture providing a factory function to create RDKit molecules from SMILES.

    Returns a function that takes a SMILES string and returns a molecule
    with 3D coordinates and hydrogens added.
    """
    def create_molecule(smiles):
        """Create a 3D molecule from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    return create_molecule


# Test data: (SMILES, name, expected_torsions, description)
ALKANE_TEST_CASES = [
    ('CCC', 'propane', 0, 'Only 3 heavy atoms, no valid A-B-C-D quartet'),
    ('CCCC', 'butane', 1, '1 central bond with 4 heavy atoms'),
    ('CCCCC', 'pentane', 2, '2 interior bonds can form C-C-C-C quartets'),
    ('CCCCCC', 'hexane', 3, '3 interior bonds with 4 heavy atoms each'),
]

BRANCHED_TEST_CASES = [
    ('CC(C)C', 'isobutane', 0, 'All bonds are terminal (CH3 groups)'),
    ('CC(C)CC', '2-methylbutane', 1, '1 torsion through the C-C backbone'),
    ('CC(C)CCC', '2-methylpentane', 2, '2 torsions along the backbone'),
]

AROMATIC_TEST_CASES = [
    ('c1ccccc1', 'benzene', 0, 'Ring bonds are not rotatable'),
    ('Cc1ccccc1', 'toluene', 0, 'CH3-Ph bond rotatable but lacks 4 heavy atoms'),
    ('CCc1ccccc1', 'ethylbenzene', 1, '1 torsion C-C-C-C (aliphatic-aromatic)'),
    ('CCCc1ccccc1', 'propylbenzene', 2, '2 torsions in the propyl chain'),
]

DRUG_TEST_CASES = [
    ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'ibuprofen', 4, '4 heavy-atom torsions in side chains'),
    ('CC(=O)Oc1ccccc1C(=O)O', 'aspirin', 3, '3 torsions (ester, aromatic, carboxyl)'),
    ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'caffeine', 0, 'Fused ring system, terminal methyls'),
]

FUNCTIONAL_GROUP_TEST_CASES = [
    ('CCCO', 'propanol', 1, '1 torsion C-C-C-O (oxygen lacks heavy neighbor)'),
    ('CCCCO', 'butanol', 2, '2 torsions in the backbone'),
    ('CCOCC', 'diethyl_ether', 2, '2 torsions C-C-O-C on each side'),
    ('CCCN', 'propylamine', 1, '1 torsion C-C-C-N (nitrogen lacks heavy neighbor)'),
    ('CCCCN', 'butylamine', 2, '2 torsions in the backbone'),
    ('CC(=O)NC', 'n-methylacetamide', 1, '1 torsion through amide bond'),
    ('CCCC(=O)O', 'pentanoic_acid', 2, '2 torsions in alkyl chain'),
]


class TestTorsionExtraction:
    """Test suite for heavy-atom torsion extraction."""

    @pytest.mark.parametrize("smiles,name,expected,description", ALKANE_TEST_CASES)
    def test_linear_alkanes(self, analyzer, molecule_factory, smiles, name, expected, description):
        """Test torsion extraction for linear alkanes."""
        mol = molecule_factory(smiles)
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == expected, (
            f"{name}: Expected {expected} torsions but got {len(torsion_info)}. "
            f"Reason: {description}"
        )

    @pytest.mark.parametrize("smiles,name,expected,description", BRANCHED_TEST_CASES)
    def test_branched_alkanes(self, analyzer, molecule_factory, smiles, name, expected, description):
        """Test torsion extraction for branched alkanes."""
        mol = molecule_factory(smiles)
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == expected, (
            f"{name}: Expected {expected} torsions but got {len(torsion_info)}. "
            f"Reason: {description}"
        )

    @pytest.mark.parametrize("smiles,name,expected,description", AROMATIC_TEST_CASES)
    def test_aromatic_compounds(self, analyzer, molecule_factory, smiles, name, expected, description):
        """Test torsion extraction for aromatic compounds."""
        mol = molecule_factory(smiles)
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == expected, (
            f"{name}: Expected {expected} torsions but got {len(torsion_info)}. "
            f"Reason: {description}"
        )

    @pytest.mark.parametrize("smiles,name,expected,description", DRUG_TEST_CASES)
    def test_drug_molecules(self, analyzer, molecule_factory, smiles, name, expected, description):
        """Test torsion extraction for common drug molecules."""
        mol = molecule_factory(smiles)
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == expected, (
            f"{name}: Expected {expected} torsions but got {len(torsion_info)}. "
            f"Reason: {description}"
        )

    @pytest.mark.parametrize("smiles,name,expected,description", FUNCTIONAL_GROUP_TEST_CASES)
    def test_functional_groups(self, analyzer, molecule_factory, smiles, name, expected, description):
        """Test torsion extraction for molecules with various functional groups."""
        mol = molecule_factory(smiles)
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == expected, (
            f"{name}: Expected {expected} torsions but got {len(torsion_info)}. "
            f"Reason: {description}"
        )


class TestTorsionProperties:
    """Test properties of extracted torsions."""

    def test_torsion_atoms_are_heavy(self, analyzer, molecule_factory):
        """Verify that all dihedral atoms are heavy atoms (not hydrogen)."""
        mol = molecule_factory('CCCCCC')  # Hexane
        torsion_info = analyzer.extract_torsion_angles(mol)

        for torsion in torsion_info:
            a, b, c, d = torsion.dihedral_atoms
            atoms = [mol.GetAtomWithIdx(idx) for idx in [a, b, c, d]]

            # All atoms should be heavy (not hydrogen)
            for atom in atoms:
                assert atom.GetAtomicNum() != 1, (
                    f"Found hydrogen in dihedral quartet {torsion.dihedral_atoms}"
                )

    def test_torsion_angle_range(self, analyzer, molecule_factory):
        """Verify that all torsion angles are in [-π, π] range."""
        mol = molecule_factory('CCCCCC')  # Hexane
        torsion_info = analyzer.extract_torsion_angles(mol)

        for torsion in torsion_info:
            assert -np.pi <= torsion.angle <= np.pi, (
                f"Torsion angle {torsion.angle} out of range [-π, π]"
            )

    def test_rotatable_bonds_match_torsions(self, analyzer, molecule_factory):
        """Verify that torsion count <= rotatable bond count."""
        mol = molecule_factory('CCCCC')  # Pentane
        rotatable_bonds = analyzer.identify_rotatable_bonds(mol)
        torsion_info = analyzer.extract_torsion_angles(mol)

        # Each torsion corresponds to a rotatable bond, but not all rotatable
        # bonds have torsions (some lack 4 heavy atoms)
        assert len(torsion_info) <= len(rotatable_bonds), (
            f"Torsions ({len(torsion_info)}) should not exceed rotatable bonds "
            f"({len(rotatable_bonds)})"
        )

    def test_bond_indices_in_torsion(self, analyzer, molecule_factory):
        """Verify that each torsion's bond indices appear in the dihedral."""
        mol = molecule_factory('CCCC')  # Butane
        torsion_info = analyzer.extract_torsion_angles(mol)

        for torsion in torsion_info:
            bond_i, bond_j = torsion.bond
            a, b, c, d = torsion.dihedral_atoms

            # The bond should be the middle bond (b-c)
            assert (b, c) == (bond_i, bond_j) or (c, b) == (bond_i, bond_j), (
                f"Bond {torsion.bond} not found in middle of dihedral "
                f"{torsion.dihedral_atoms}"
            )


class TestIbuprofenRegression:
    """Regression test for the original Ibuprofen bug."""

    def test_ibuprofen_four_torsions(self, analyzer, molecule_factory):
        """
        Regression test: Ibuprofen should have exactly 4 heavy-atom torsions.

        This was the original bug that was fixed - it was returning 8 torsions
        because it was including H-C-C-H type dihedrals.
        """
        mol = molecule_factory('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == 4, (
            f"Ibuprofen should have 4 heavy-atom torsions, got {len(torsion_info)}. "
            f"This is a regression of the original bug!"
        )

    def test_ibuprofen_all_heavy_atoms(self, analyzer, molecule_factory):
        """Verify that all Ibuprofen torsions use only heavy atoms."""
        mol = molecule_factory('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
        torsion_info = analyzer.extract_torsion_angles(mol)

        for i, torsion in enumerate(torsion_info, 1):
            a, b, c, d = torsion.dihedral_atoms
            atoms = [mol.GetAtomWithIdx(idx) for idx in [a, b, c, d]]
            symbols = [atom.GetSymbol() for atom in atoms]

            # None should be hydrogen
            assert 'H' not in symbols, (
                f"Torsion {i} contains hydrogen: {symbols} at indices {torsion.dihedral_atoms}"
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_rotatable_bonds(self, analyzer, molecule_factory):
        """Test molecules with no rotatable bonds."""
        mol = molecule_factory('c1ccccc1')  # Benzene
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == 0, "Benzene should have no rotatable bonds"

    def test_terminal_bonds_excluded(self, analyzer, molecule_factory):
        """Test that terminal bonds (like CH3-CH3) don't create torsions."""
        mol = molecule_factory('CC')  # Ethane
        torsion_info = analyzer.extract_torsion_angles(mol)

        assert len(torsion_info) == 0, (
            "Ethane (terminal bond) should have no torsions"
        )

    def test_minimum_size_for_torsion(self, analyzer, molecule_factory):
        """Test that minimum 4 heavy atoms are needed for a torsion."""
        # C-C-C (3 heavy atoms) should have 0 torsions
        mol = molecule_factory('CCC')
        torsion_info = analyzer.extract_torsion_angles(mol)
        assert len(torsion_info) == 0, "Propane (3 heavy atoms) should have no torsions"

        # C-C-C-C (4 heavy atoms) should have 1 torsion
        mol = molecule_factory('CCCC')
        torsion_info = analyzer.extract_torsion_angles(mol)
        assert len(torsion_info) == 1, "Butane (4 heavy atoms) should have 1 torsion"


if __name__ == '__main__':
    # Allow running with: python test_torsion_fix.py
    pytest.main([__file__, '-v', '-s'])
