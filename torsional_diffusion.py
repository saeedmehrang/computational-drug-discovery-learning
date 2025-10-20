"""
Torsional Diffusion Model for Molecular Conformation Generation

This module implements a SIMPLIFIED torsion-based diffusion model using E(n) Equivariant
Graph Neural Networks (EGNN) for molecular conformation generation. The approach is
inspired by state-of-the-art methods from 2024-2025 research.

IMPORTANT - EDUCATIONAL SIMPLIFICATION:
This implementation uses a SIMPLIFIED approach to torsional diffusion compared to the
original Jing et al. (2022) paper. Specifically:

1. **Simplified Periodic Handling**: We use standard DDPM with post-hoc angle wrapping
   via atan2(sin(θ), cos(θ)), rather than proper diffusion on the hypertorus manifold.

2. **Not True Hypertorus Diffusion**: The original paper uses rigorous score matching
   on the torus with wrapped normal or von Mises distributions. Our implementation
   applies Gaussian noise and wraps the result, which is an approximation.

3. **Performance Tradeoff**: This simplification makes the code easier to understand
   but will NOT match the performance of the original paper. For production use,
   implement proper wrapped normal kernels or von Mises-based diffusion.

For educational purposes, this implementation demonstrates the core concepts while
remaining accessible. For research or production, refer to the official implementation:
https://github.com/gcorso/torsional-diffusion

Key concepts:
- Torsional diffusion: Diffuse only rotatable dihedral angles (not full 3D coordinates)
- EGNN backbone: SE(3)-equivariant message passing for molecular geometry
- Circular diffusion: Handle periodic nature of torsion angles [-π, π]

References:
- Jing, Bowen, et al. "Torsional diffusion for molecular conformer generation."
  Advances in neural information processing systems 35 (2022): 24240-24253.
- Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling. "E (n) equivariant graph neural networks."
  International conference on machine learning. PMLR, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TorsionInfo:
    """
    Information about a rotatable torsion angle in a molecule.

    Attributes:
        bond: Tuple of (atom_i, atom_j) indices defining the rotatable bond
        dihedral_atoms: Tuple of (a, i, j, b) indices defining the dihedral angle
        angle: Current torsion angle in radians [-π, π]
    """
    bond: Tuple[int, int]
    dihedral_atoms: Tuple[int, int, int, int]
    angle: float


# ============================================================================
# Molecular Torsion Analysis
# ============================================================================

class MolecularTorsionAnalyzer:
    """
    Identifies rotatable bonds and extracts torsion angles from molecules.

    This class implements the core logic for:
    1. Identifying which bonds in a molecule can rotate (single, non-ring, non-terminal)
    2. Computing dihedral angles from 3D coordinates
    3. Extracting all torsion information needed for diffusion
    """

    @staticmethod
    def identify_rotatable_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
        """
        Find all rotatable bonds in a molecule.

        A bond is considered rotatable if:
        1. It's a single bond (not double or triple)
        2. It's not in a ring structure
        3. It's not terminal (both atoms have degree >= 2)
        4. Neither atom is hydrogen

        Args:
            mol: RDKit molecule object with defined connectivity

        Returns:
            List of tuples (atom_idx1, atom_idx2) for each rotatable bond

        Example:
            For ethane (C-C), there is 1 rotatable bond
            For benzene, there are 0 rotatable bonds (all in ring)
        """
        rotatable_bonds = []

        # Get ring information - bonds in rings are typically not rotatable
        ring_info = mol.GetRingInfo()
        bonds_in_rings = set()
        for ring in ring_info.BondRings():
            bonds_in_rings.update(ring)

        # Check each bond in the molecule
        for bond in mol.GetBonds():
            bond_idx = bond.GetIdx()
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # Rule 1: Must be single bond (not double/triple)
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            # Rule 2: Not in a ring (rings are rigid)
            if bond_idx in bonds_in_rings:
                continue

            # Rule 3: Not a terminal bond (both atoms need degree >= 2)
            # Terminal bonds like CH3 have no rotational conformers
            if atom1.GetDegree() < 2 or atom2.GetDegree() < 2:
                continue

            # Rule 4: Skip bonds to hydrogen (handled implicitly in coordinates)
            if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
                continue

            rotatable_bonds.append((atom1.GetIdx(), atom2.GetIdx()))

        return rotatable_bonds

    @staticmethod
    def get_dihedral_atoms(mol: Chem.Mol, bond_atoms: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        For a rotatable bond (i-j), find the 4 atoms (a-i-j-b) defining the dihedral angle.

        The dihedral angle is defined by 4 atoms in sequence:
        - atom_a: heavy-atom neighbor of i (not j, not hydrogen)
        - atom_i: first atom of rotatable bond
        - atom_j: second atom of rotatable bond
        - atom_b: heavy-atom neighbor of j (not i, not hydrogen)

        This ensures we get the canonical heavy-atom torsion angle, which is what's
        used in most molecular modeling applications and torsional diffusion models.

        Args:
            mol: RDKit molecule object
            bond_atoms: Tuple (atom_i_idx, atom_j_idx) of the rotatable bond

        Returns:
            Tuple (a, i, j, b) of atom indices, or None if heavy atoms cannot be found

        Example:
            For a bond C2-C3 in butane: H-C1-C2-C3-C4-H
            Returns (C1, C2, C3, C4) indices (not hydrogens)
        """
        atom_i_idx, atom_j_idx = bond_atoms
        atom_i = mol.GetAtomWithIdx(atom_i_idx)
        atom_j = mol.GetAtomWithIdx(atom_j_idx)

        # Find a HEAVY-ATOM neighbor of atom_i (excluding atom_j and hydrogens)
        atom_a = None
        for neighbor in atom_i.GetNeighbors():
            if neighbor.GetIdx() != atom_j_idx and neighbor.GetAtomicNum() != 1:
                atom_a = neighbor.GetIdx()
                break

        # Find a HEAVY-ATOM neighbor of atom_j (excluding atom_i and hydrogens)
        atom_b = None
        for neighbor in atom_j.GetNeighbors():
            if neighbor.GetIdx() != atom_i_idx and neighbor.GetAtomicNum() != 1:
                atom_b = neighbor.GetIdx()
                break

        if atom_a is None or atom_b is None:
            return None

        return (atom_a, atom_i_idx, atom_j_idx, atom_b)

    @staticmethod
    def calculate_dihedral_angle(coords: np.ndarray, atom_indices: Tuple[int, int, int, int]) -> float:
        """
        Calculate dihedral (torsion) angle from 4 atom coordinates.

        The dihedral angle is the angle between two planes:
        - Plane 1: defined by atoms (a, b, c)
        - Plane 2: defined by atoms (b, c, d)

        Uses the standard formula: atan2(m1·n2, n1·n2)
        where n1, n2 are plane normals and m1 = n1 x b2_unit

        Args:
            coords: Nx3 numpy array of 3D coordinates for all atoms
            atom_indices: Tuple (a, b, c, d) of atom indices defining the dihedral

        Returns:
            Dihedral angle in radians, range [-π, π]

        Note:
            This uses the IUPAC convention where 0° is cis and ±180° is trans
        """
        a, b, c, d = atom_indices

        # Get 3D coordinates for each atom
        p0 = coords[a]
        p1 = coords[b]
        p2 = coords[c]
        p3 = coords[d]

        # Calculate bond vectors
        b1 = p1 - p0  # Vector from a to b
        b2 = p2 - p1  # Vector from b to c (the rotatable bond)
        b3 = p3 - p2  # Vector from c to d

        # Calculate normal vectors to the two planes
        n1 = np.cross(b1, b2)  # Normal to plane (a, b, c)
        n2 = np.cross(b2, b3)  # Normal to plane (b, c, d)

        # Normalize vectors (avoid division by zero)
        n1 = n1 / (np.linalg.norm(n1) + 1e-8)
        n2 = n2 / (np.linalg.norm(n2) + 1e-8)
        b2_unit = b2 / (np.linalg.norm(b2) + 1e-8)

        # Calculate dihedral angle using atan2 for correct quadrant
        m1 = np.cross(n1, b2_unit)
        x = np.dot(n1, n2)      # cos(angle)
        y = np.dot(m1, n2)      # sin(angle)

        angle = np.arctan2(y, x)
        return angle

    @staticmethod
    def extract_torsion_angles(mol: Chem.Mol, conf_id: int = -1) -> List[TorsionInfo]:
        """
        Extract all torsion angles from a molecule conformation.

        This is the main function that combines:
        1. Identification of rotatable bonds
        2. Finding dihedral atom quartets
        3. Computing actual angle values

        Args:
            mol: RDKit molecule with 3D coordinates embedded
            conf_id: Conformation ID to use (default: -1 for most recent)

        Returns:
            torsion_info: List of TorsionInfo objects with detailed info

        Example:
            mol = Chem.AddHs(Chem.MolFromSmiles('CCC'))
            AllChem.EmbedMolecule(mol)
            info = extract_torsion_angles(mol)
            # For propane, returns 1 torsion angle
        """
        conf = mol.GetConformer(conf_id)
        coords = conf.GetPositions()

        # Step 1: Find all rotatable bonds
        rotatable_bonds = MolecularTorsionAnalyzer.identify_rotatable_bonds(mol)

        torsion_info = []

        # Step 2: For each rotatable bond, compute its torsion angle
        for bond in rotatable_bonds:
            # Get the 4 atoms (a-i-j-b) defining the dihedral
            dihedral_atoms = MolecularTorsionAnalyzer.get_dihedral_atoms(mol, bond)

            if dihedral_atoms is None:
                continue

            # Calculate the actual angle value
            angle = MolecularTorsionAnalyzer.calculate_dihedral_angle(coords, dihedral_atoms)

            torsion_info.append(TorsionInfo(
                bond=bond,
                dihedral_atoms=dihedral_atoms,
                angle=angle
            ))

        return torsion_info


# ============================================================================
# EGNN (E(n) Equivariant Graph Neural Network)
# ============================================================================

class EGNN_Layer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network Layer.

    Maintains equivariance by separating invariant (features, distance)
    and equivariant (coordinates, displacement vectors) quantities.

    Refactored to use standard PyG tuple message passing.
    """

    def __init__(
        self,
        in_node_features: int,
        hidden_dim: int,
        out_node_features: int,
        residual: bool = True  # Add option for residual connection
    ):
        # We aggregate feature messages and coordinate messages by summation ('add')
        super().__init__(aggr='add')
        self.residual = residual

        # Edge model: Φ_e (message function for features)
        # Input: [h_i, h_j, ||x_i - x_j||^2]
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_features * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node model: Φ_h (feature update function)
        # Input: [h_i, aggregated_feature_messages]
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_features + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_node_features)
        )

        # Coordinate model: Φ_x (position update weight)
        # Outputs scalar weights (invariant) based on feature messages
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)  # Scalar output
        )

    def forward(
        self,
        h: torch.Tensor,           # Node features [num_nodes, in_node_features]
        pos: torch.Tensor,         # Node positions [num_nodes, 3]
        edge_index: torch.Tensor   # Edge indices [2, num_edges]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates messages for feature and coordinate updates simultaneously.
        """
        # propagate returns a tuple: (aggregated_feature_messages, aggregated_coord_updates)
        agg_h, agg_pos_delta = self.propagate(
            edge_index,
            h=h,
            pos=pos,
            size=(h.size(0), h.size(0))
        )

        # 1. Update positions (Equivariant)
        pos_updated = pos + agg_pos_delta

        # 2. Update node features (Invariant)
        h_updated = self.update_h(h, agg_h)

        return h_updated, pos_updated

    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs messages for features (invariant) and coordinates (equivariant).
        """
        # Compute relative positions (equivariant)
        pos_diff = pos_i - pos_j  # [num_edges, 3]

        # Compute squared distances (invariant)
        dist_squared = torch.sum(pos_diff ** 2, dim=-1, keepdim=True)  # [num_edges, 1]

        # 1. Feature Message (Invariant)
        edge_input = torch.cat([h_i, h_j, dist_squared], dim=-1)
        edge_message = self.edge_mlp(edge_input)  # [num_edges, hidden_dim]

        # 2. Coordinate Message (Equivariant)
        # Compute scalar weight (invariant)
        coord_weight = self.coord_mlp(edge_message)  # [num_edges, 1]

        # Multiply scalar weight by displacement vector (Equivariant!)
        coord_message = pos_diff * coord_weight  # [num_edges, 3]

        # RETURN A TUPLE: PyG will aggregate both tensors separately
        return edge_message, coord_message

    # Note: When message returns a tuple, PyG automatically aggregates them
    # separately based on the 'aggr' (summation in this case).
    # We rename 'update' to 'update_h' for clarity and handle coordinates
    # directly in the forward pass after aggregation.
    def update_h(
        self,
        h: torch.Tensor,           # Original features
        agg_h: torch.Tensor        # Aggregated feature messages
    ) -> torch.Tensor:
        """
        Update node features using aggregated messages.
        """
        node_input = torch.cat([h, agg_h], dim=-1)
        h_new = self.node_mlp(node_input)

        # Apply residual connection if enabled and dimensions match
        if self.residual and h.shape[-1] == h_new.shape[-1]:
            return h + h_new
        else:
            return h_new


# The EGNN wrapper class remains correct and robust.
class EGNN(nn.Module):
    """
    Multi-layer E(n) Equivariant Graph Neural Network.
    """

    def __init__(
        self,
        in_node_features: int,
        hidden_dim: int,
        out_node_features: int,
        num_layers: int = 3,
        residual: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer (in_node_features -> hidden_dim)
        self.layers.append(
            EGNN_Layer(in_node_features, hidden_dim, hidden_dim, residual)
        )

        # Middle layers (hidden_dim -> hidden_dim)
        for _ in range(num_layers - 2):
            self.layers.append(
                EGNN_Layer(hidden_dim, hidden_dim, hidden_dim, residual)
            )

        # Final layer (hidden_dim -> out_node_features)
        self.layers.append(
            EGNN_Layer(hidden_dim, hidden_dim, out_node_features, residual)
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        return h, pos

# ============================================================================
# Torsion Angle Denoising Network (EGNN-based)
# ============================================================================

class TorsionDenoiser(nn.Module):
    """
    Neural network for predicting noise in torsion angles using EGNN.
    
    Revised for vectorized time-embedding and simplified torsion input handling.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 128,
        num_egnn_layers: int = 3
    ):
        """
        Initialize the torsion angle denoising network.
        
        Architecture components:
            - Time MLP: Processes sinusoidal timestep embeddings
            - Node embedding: Projects atom features to hidden dimension
            - EGNN backbone: E(n)-equivariant message passing for geometry processing
            - Torsion MLP: Predicts per-torsion noise from local atomic context
        
        Args:
            atom_feature_dim: Dimension of input atomic features
                Common values: 64-128 (depends on featurization scheme)
            hidden_dim: Hidden layer dimension for all networks
                Larger values increase capacity but slow training
                Recommended: 128-256
            num_egnn_layers: Number of EGNN layers to stack
                More layers = larger receptive field but slower
                Recommended: 3-5 for molecules with <100 atoms
        
        Example:
            >>> model = TorsionDenoiser(
            ...     atom_feature_dim=74,  # e.g., one-hot element + 6 extra features
            ...     hidden_dim=128,
            ...     num_egnn_layers=4
            ... )
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initial node feature embedding
        # Correctly uses the input feature dimension
        self.node_embedding = nn.Linear(atom_feature_dim, hidden_dim)

        # EGNN backbone
        self.egnn = EGNN(
            in_node_features=hidden_dim,
            hidden_dim=hidden_dim,
            out_node_features=hidden_dim,
            num_layers=num_egnn_layers
        )

        # Torsion-level prediction head
        self.torsion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),  # 4 atoms + sin/cos
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Predict noise for this torsion
        )

    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embeddings for diffusion timesteps.
        
        Uses Transformer-style positional encodings to embed continuous timestep values:
            PE(t, 2i) = sin(t / 10000^(2i/d))
            PE(t, 2i+1) = cos(t / 10000^(2i/d))
        
        This encoding allows the network to learn temporal patterns in the denoising
        process and distinguish between early (high noise) and late (low noise) timesteps.
        
        Args:
            timesteps: Diffusion timestep values [batch_size]
                Typically in range [0, num_diffusion_steps]
        
        Returns:
            Time embeddings [batch_size, hidden_dim]
                High-dimensional representation of timesteps for conditioning the network
        
        Note:
            Handles odd hidden_dim by padding with zeros to avoid dimension mismatch.
        """
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        
        # Handle case where hidden_dim is odd
        if self.hidden_dim % 2 != 0:
            pad = torch.zeros(timesteps.shape[0], 1, device=timesteps.device)
            emb = torch.cat([torch.sin(emb), torch.cos(emb), pad], dim=-1)
        else:
             emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return self.time_mlp(emb)


    def forward(
        self,
        torsions_t: torch.Tensor,        # [total_torsions_in_batch]
        t: torch.Tensor,                 # [batch_size]
        node_features: torch.Tensor,     # [total_atoms, feature_dim]
        pos: torch.Tensor,               # [total_atoms, 3]
        edge_index: torch.Tensor,        # [2, num_edges]
        torsion_to_atoms: torch.Tensor,  # [total_torsions_in_batch, 4]
        batch: torch.Tensor,             # [total_atoms]
        torsion_batch_idx: torch.Tensor, # [total_torsions_in_batch]
    ) -> torch.Tensor:
        """
        Predict noise in torsion angles using geometry-aware message passing.
        
        Pipeline:
            1. Embed timestep t into high-dimensional representation
            2. Initialize node features and broadcast time embeddings to all atoms
            3. Run EGNN to refine features using 3D molecular geometry
            4. Extract features of 4 atoms defining each torsion angle
            5. Combine with circular encoding (sin/cos) of current torsion values
            6. Predict noise for each torsion independently
        
        Args:
            torsions_t: Noisy torsion angles at timestep t [total_torsions_in_batch]
                Flattened across all molecules in batch (in radians)
            t: Diffusion timesteps [batch_size]
                One timestep value per molecule
            node_features: Atomic features [total_atoms, feature_dim]
                Features like atom type, charge, hybridization, etc.
            pos: 3D atomic coordinates [total_atoms, 3]
                Current positions (Angstroms)
            edge_index: Molecular graph connectivity [2, num_edges]
                Edge list in PyG format: [source_nodes; target_nodes]
            torsion_to_atoms: Atom indices defining torsions [total_torsions_in_batch, 4]
                Each row contains [atom_i, atom_j, atom_k, atom_l] for torsion i-j-k-l
            batch: Batch assignment for atoms [total_atoms]
                Maps each atom to its molecule index
            torsion_batch_idx: Batch assignment for torsions [total_torsions_in_batch]
                Maps each torsion to its molecule index
        
        Returns:
            noise_pred: Predicted noise for each torsion [total_torsions_in_batch]
                Values to subtract from torsions_t to denoise (in radians)
        
        Note:
            Uses flat batching to naturally handle molecules with different numbers
            of torsions. The caller is responsible for gathering predictions by
            molecule using torsion_batch_idx if needed.
        """
        # 1. Time embedding and feature initialization
        time_embed = self.get_time_embedding(t)  # [batch, hidden_dim]
        h = self.node_embedding(node_features)   # [total_atoms, hidden_dim]

        # 2. Vectorized Time Embedding Broadcast (Fix for Issue B)
        # time_embed[batch] gathers the correct time embedding for each atom
        time_embed_per_node = time_embed[batch]
        h = h + time_embed_per_node              # [total_atoms, hidden_dim]

        # 3. Run EGNN
        h, pos_updated = self.egnn(h, pos, edge_index) # [total_atoms, hidden_dim]

        # 4. Vectorized Torsion Prediction
        
        # Gather features for all 4 atoms defining each torsion
        h_0 = h[torsion_to_atoms[:, 0]]
        h_1 = h[torsion_to_atoms[:, 1]]
        h_2 = h[torsion_to_atoms[:, 2]]
        h_3 = h[torsion_to_atoms[:, 3]]
        atom_features = torch.cat([h_0, h_1, h_2, h_3], dim=-1) # [total_torsions, hidden_dim*4]

        # Circular encoding for all torsions
        circular_enc = torch.stack([
            torch.sin(torsions_t),
            torch.cos(torsions_t)
        ], dim=-1) # [total_torsions, 2]

        # Combine and predict
        torsion_input = torch.cat([atom_features, circular_enc], dim=-1)
        noise_pred_flat = self.torsion_mlp(torsion_input).squeeze(-1) # [total_torsions]

        # 5. Reshape Output (Fix for Issue A)
        # If the output needs to be [batch, max_torsions], the caller must handle 
        # padding, which is complex and often unnecessary for loss calculation.
        # It's safest to return the flat prediction and rely on the loss function
        # to use the `torsion_batch_idx` if needed.
        
        # We return the flat tensor [total_torsions_in_batch]
        return noise_pred_flat
    

# ============================================================================
# Torsional Diffusion Model
# ============================================================================

class TorsionalDiffusionModel(nn.Module):
    """
    Complete torsional diffusion model for molecular conformation generation.
    
    This implements the DDPM (Denoising Diffusion Probabilistic Model) framework
    specifically for molecular torsion angles.
    
    Key components:
    1. Noise schedule: Cosine schedule for β_t (better than linear)
    2. Forward process: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
    3. Reverse process: p_θ(x_{t-1} | x_t) using EGNN denoiser
    4. Circular wrapping: Handle periodic nature of angles
    
    Args:
        atom_feature_dim: Dimension of input atomic features
        hidden_dim: Hidden dimension for neural networks
        num_timesteps: Number of diffusion steps (typically 1000)
        num_egnn_layers: Number of EGNN layers in denoiser
    
    Training:
        loss = MSE(ε, ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t))
        where ε ~ N(0, I) is the added noise
    
    Sampling:
        Start from x_T ~ N(0, I)
        For t = T to 1:
            x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t z
        Return x_0
    """
    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 128,
        num_timesteps: int = 1000,
        num_egnn_layers: int = 3
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        
        # Noise schedule parameters
        self.register_buffer('betas', self.cosine_schedule(num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        # Denoising network ε_θ
        self.denoiser = TorsionDenoiser(
            atom_feature_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_egnn_layers=num_egnn_layers
        )
    
    def cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule for diffusion.
        
        Better than linear schedule - provides:
        - Slower noise addition at start (preserve structure)
        - Faster noise at end (reach noise quickly)
        
        Formula: ᾱ_t = cos²((t/T + s)/(1 + s) · π/2)
        
        Args:
            timesteps: Total number of diffusion steps
            s: Small offset for numerical stability
        
        Returns:
            betas: Noise schedule [num_timesteps]
        
        Reference:
            Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise_to_torsions(
        self,
        torsions_0: torch.Tensor,  # [total_torsions]
        t: torch.Tensor            # [total_torsions]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to clean torsion angles.
        
        Implements: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        where ε ~ N(0, I)
        
        EDUCATIONAL SIMPLIFICATION NOTE:
        This uses standard Gaussian noise + post-hoc wrapping, which is a simplification.
        The original Jing et al. (2022) paper uses proper diffusion on the hypertorus
        with wrapped normal or von Mises distributions for rigorous circular statistics.
        
        Args:
            torsions_0: Clean torsion angles [total_torsions] (radians)
            t: Timestep for each torsion [total_torsions]
        
        Returns:
            torsions_t: Noisy torsion angles [total_torsions]
            noise: The noise that was added [total_torsions]
        """
        # Sample random noise
        noise = torch.randn_like(torsions_0)
        
        # FIX 1: Proper broadcasting with explicit reshape
        # Get alpha_bar values for each timestep
        alpha_bar_t = self.alpha_bars[t]  # [total_torsions]
        
        # Compute coefficients
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        # Standard DDPM forward process
        torsions_t = sqrt_alpha_bar_t * torsions_0 + sqrt_one_minus_alpha_bar_t * noise
        
        # Wrap angles to [-π, π]
        torsions_t = torch.atan2(torch.sin(torsions_t), torch.cos(torsions_t))
        
        return torsions_t, noise
    
    def training_step(
        self,
        clean_torsions_flat: torch.Tensor,  # [total_torsions]
        node_features: torch.Tensor,         # [total_atoms, feature_dim]
        pos: torch.Tensor,                   # [total_atoms, 3]
        edge_index: torch.Tensor,            # [2, num_edges]
        torsion_to_atoms: torch.Tensor,      # [total_torsions, 4]
        batch: torch.Tensor,                 # [total_atoms]
        torsion_batch_idx: torch.Tensor      # [total_torsions]
    ) -> torch.Tensor:
        """
        Single training step for torsional diffusion.
        
        DDPM Training Objective:
            L_simple = E_{x_0, ε, t} [||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]
        
        Where:
            - x_0: clean torsion angles
            - ε ~ N(0, I): random noise
            - t ~ Uniform(0, T): random timestep
            - ε_θ: neural network (EGNN denoiser)
        
        Batching Strategy:
            Uses PyTorch Geometric's flat batching:
            - All torsions from all molecules concatenated: [total_torsions]
            - All atoms from all molecules concatenated: [total_atoms]
            - Mappings track which molecule each torsion/atom belongs to
        
        Args:
            clean_torsions_flat: Ground truth angles [total_torsions] (radians, in [-π, π])
            node_features: Atom features [total_atoms, feature_dim]
            pos: 3D coordinates [total_atoms, 3]
            edge_index: Molecular bonds [2, num_edges]
            torsion_to_atoms: Defines each torsion [total_torsions, 4]
            batch: Maps atoms to molecules [total_atoms]
            torsion_batch_idx: Maps torsions to molecules [total_torsions]
        
        Returns:
            loss: MSE between true noise and predicted noise (scalar)
        """
        # FIX 4: Add validation
        mol_batch_size = batch.max().item() + 1
        total_torsions = clean_torsions_flat.shape[0]
        device = clean_torsions_flat.device
        
        assert torsion_batch_idx.max().item() + 1 == mol_batch_size, \
            "Mismatch between atom batch and torsion batch sizes"
        assert len(torsion_batch_idx) == total_torsions, \
            "torsion_batch_idx length must match total_torsions"
        
        # Sample random timesteps per MOLECULE
        t_mol = torch.randint(0, self.num_timesteps, (mol_batch_size,), device=device)
        
        # Broadcast timestep for each TORSION using torsion_batch_idx
        # This maps t_mol[i] to every torsion belonging to molecule i
        t = t_mol[torsion_batch_idx]  # [total_torsions]
        
        # Add noise to torsions (using flat tensors)
        noisy_torsions, true_noise = self.add_noise_to_torsions(clean_torsions_flat, t)
        
        # Predict noise using EGNN denoiser
        predicted_noise = self.denoiser(
            torsions_t=noisy_torsions,
            t=t_mol,  # Pass molecular timesteps [batch_size]
            node_features=node_features,
            pos=pos,
            edge_index=edge_index,
            torsion_to_atoms=torsion_to_atoms,
            batch=batch,
            torsion_batch_idx=torsion_batch_idx
        )
        
        # Compute loss (MSE on flat noise tensors)
        loss = F.mse_loss(predicted_noise, true_noise)
        
        return loss
    
    @torch.no_grad()
    def generate_torsions(
        self,
        node_features: torch.Tensor,       # [num_atoms, feature_dim]
        pos: torch.Tensor,                 # [num_atoms, 3]
        edge_index: torch.Tensor,          # [2, num_edges]
        torsion_to_atoms: torch.Tensor,    # [total_torsions, 4]
        batch: torch.Tensor,               # [num_atoms]
        torsion_batch_idx: torch.Tensor,   # [total_torsions]
        total_torsions: int
    ) -> torch.Tensor:
        """
        Generates new torsion angles via the reverse diffusion process (sampling).
        
        The process starts from pure Gaussian noise (x_T) and iteratively denoises
        the sample over 'num_timesteps' steps using the EGNN denoiser and the
        pre-defined noise schedule.
        
        Algorithm (DDPM Sampling):
        1. Start with random noise x_T ~ N(0, I) (scaled).
        2. For t = T-1 down to 0:
            a. Predict noise ε_θ(x_t, t) using the denoiser.
            b. Calculate the mean μ_θ using the DDPM formula.
            c. Sample x_{t-1} = μ_θ + σ_t z (adding noise except at t=0).
            d. Wrap angles to [-π, π].
        
        Args:
            node_features: Atomic features for the single molecule being generated.
                           Shape: [num_atoms, feature_dim]
            pos: 3D coordinates. Shape: [num_atoms, 3]
            edge_index: Graph connectivity. Shape: [2, num_edges]
            torsion_to_atoms: Torsion definitions. Shape: [total_torsions, 4]
            batch: Atom batch assignment (typically all zeros for a single molecule).
                   Shape: [num_atoms]
            torsion_batch_idx: Torsion batch assignment (typically all zeros).
                               Shape: [total_torsions]
            total_torsions: The exact number of torsions to be generated.
        
        Returns:
            Generated torsion angles (x_0). Shape: [total_torsions]
        """
        # FIX 4: Add validation assertions
        assert batch.max().item() == 0, \
            "Generation only supports single molecule (batch_size=1). All batch indices must be 0."
        assert torsion_batch_idx.max().item() == 0, \
            "All torsions must belong to molecule 0 for generation."
        assert len(torsion_batch_idx) == total_torsions, \
            f"Mismatch in torsion count: torsion_batch_idx has {len(torsion_batch_idx)} elements, expected {total_torsions}"
        
        device = next(self.parameters()).device
        
        # FIX 2: Remove unused line
        # Start from random noise (scaled by π for angle range)
        torsions = torch.randn(total_torsions, device=device) * math.pi  # [total_torsions]
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            # Single timestep tensor for the molecule
            t_tensor = torch.tensor([t], device=device)  # [1]
            
            # Predict noise at this timestep
            predicted_noise = self.denoiser(
                torsions_t=torsions,
                t=t_tensor,
                node_features=node_features,
                pos=pos,
                edge_index=edge_index,
                torsion_to_atoms=torsion_to_atoms,
                batch=batch,
                torsion_batch_idx=torsion_batch_idx
            )
            
            # Get schedule parameters
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            
            # Compute denoised mean using DDPM formula
            # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
            torsions = (1 / torch.sqrt(alpha_t)) * (
                torsions - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            )
            
            # Wrap to [-π, π]
            torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))
            
            # Add noise for all steps except the last
            if t > 0:
                noise = torch.randn_like(torsions)
                
                # FIX 3: Use correct posterior variance
                # Posterior variance: σ_t² = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                alpha_bar_t_prev = self.alpha_bars[t - 1]
                
                # Compute posterior variance (clipped for numerical stability)
                posterior_variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
                posterior_variance = torch.clamp(posterior_variance, min=1e-20)
                
                sigma_t = torch.sqrt(posterior_variance)
                torsions = torsions + sigma_t * noise
                
                # Wrap again after adding noise
                torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))
        
        return torsions  # [total_torsions]

# ============================================================================
# Data Processing Functions
# ============================================================================

def smiles_to_graph_data(
    smiles: str,
    add_hydrogens: bool = True
) -> Tuple[Data, Chem.Mol, List[TorsionInfo]]:
    """
    Convert SMILES string to PyTorch Geometric Data object with torsion info.

    This function:
    1. Parses SMILES into RDKit molecule
    2. Optionally adds hydrogens (important for complete structure)
    3. Generates 3D coordinates
    4. Extracts torsion angles
    5. Creates graph representation for EGNN

    Args:
        smiles: SMILES string representation of molecule
        add_hydrogens: Whether to add explicit hydrogens (recommended)

    Returns:
        data: PyTorch Geometric Data object with:
            - x: Node features [num_atoms, feature_dim]
            - edge_index: Graph connectivity [2, num_edges]
            - pos: 3D coordinates [num_atoms, 3]
            - torsion_angles: Ground truth torsions [num_torsions]
            - torsion_to_atoms: Atom indices [num_torsions, 4]
        mol: RDKit molecule object
        torsion_info: List of TorsionInfo objects
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens if requested
    if add_hydrogens:
        mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    # FIX: Removed fixed randomSeed to allow diverse conformer generation
    # For training diversity, we want different conformations each time
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)  # Energy minimization

    # Extract atomic features
    num_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom in mol.GetAtoms():
        # Simple one-hot encoding of atomic number (up to 128 elements)
        atomic_num = atom.GetAtomicNum()
        feat = torch.zeros(128)
        feat[atomic_num] = 1.0
        atom_features.append(feat)

    x = torch.stack(atom_features)  # [num_atoms, 128]

    # Extract 3D coordinates
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # [num_atoms, 3]

    # Build edge index (fully connected graph, or use RDKit bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Extract torsion angles
    analyzer = MolecularTorsionAnalyzer()
    torsion_info = analyzer.extract_torsion_angles(mol)

    # Create torsion-to-atoms mapping
    torsion_angles = []
    torsion_to_atoms = []
    for info in torsion_info:
        torsion_to_atoms.append(list(info.dihedral_atoms))
        torsion_angles.append(info.angle)

    torsion_to_atoms = torch.tensor(torsion_to_atoms, dtype=torch.long)
    torsion_angles = torch.tensor(torsion_angles, dtype=torch.float32)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        torsion_angles=torsion_angles,
        torsion_to_atoms=torsion_to_atoms
    )

    return data, mol, torsion_info


def prepare_dataset(smiles_list: List[str]) -> List[Tuple[Data, Chem.Mol, str]]:
    """
    Prepare a dataset of molecules from SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of (data, mol, smiles) tuples
    """
    dataset = []

    for smiles in smiles_list:
        try:
            data, mol, torsion_info = smiles_to_graph_data(smiles, add_hydrogens=True)
            dataset.append((data, mol, smiles))
            print(f"✓ Processed: {smiles} ({len(torsion_info)} torsions)")
        except Exception as e:
            print(f"✗ Failed: {smiles} - {e}")

    return dataset


# ============================================================================
# Training and Evaluation
# ============================================================================
def train_torsional_diffusion(
    dataset: List[Tuple[Data, Chem.Mol, str]],
    atom_feature_dim: int= 128,
    num_epochs: int = 100,
    lr: float = 1e-4,
    hidden_dim: int = 64,
    num_timesteps: int = 1000
):
    """
    Train the torsional diffusion model on a dataset.
    
    Args:
        dataset: List of (data, mol, smiles) tuples
        num_epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension for networks
        num_timesteps: Number of diffusion steps
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING TORSIONAL DIFFUSION MODEL")
    print("=" * 70)
    
    # Create model
    model = TorsionalDiffusionModel(
        atom_feature_dim=atom_feature_dim,  # FIX: Was missing the value
        hidden_dim=hidden_dim,
        num_timesteps=num_timesteps,
        num_egnn_layers=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_processed = 0  # Track actual number of molecules processed
        
        for data, mol, smiles in dataset:
            # Skip molecules with no torsions
            if len(data.torsion_angles) == 0:
                continue
            
            optimizer.zero_grad()
            
            # FIX 2: Prepare batch indices correctly for flat batching
            # Create atom batch indices (all zeros for single molecule)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            
            # FIX 3: Create torsion batch indices (all zeros for single molecule)
            num_torsions = len(data.torsion_angles)
            torsion_batch_idx = torch.zeros(num_torsions, dtype=torch.long)
            
            # FIX 4: Use flat torsion tensor (remove unsqueeze)
            # training_step expects [total_torsions], not [1, num_torsions]
            clean_torsions_flat = data.torsion_angles  # [num_torsions]
            
            # Training step with corrected arguments
            loss = model.training_step(
                clean_torsions_flat=clean_torsions_flat,  # FIX: Renamed parameter
                node_features=data.x,
                pos=data.pos,
                edge_index=data.edge_index,
                torsion_to_atoms=data.torsion_to_atoms,
                batch=batch,
                torsion_batch_idx=torsion_batch_idx  # FIX: Added missing parameter
            )
            
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_processed += 1
        
        # FIX 5: Use num_processed instead of len(dataset) for accurate average
        # Some molecules might be skipped (no torsions)
        if num_processed > 0:
            avg_loss = total_loss / num_processed
        else:
            avg_loss = 0.0
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Molecules: {num_processed}")
    
    print("\n✓ Training completed!")
    return model


@torch.no_grad()
def generate_conformer(
    model: TorsionalDiffusionModel,
    data: Data,
    mol: Chem.Mol
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate a new molecular conformer using the trained diffusion model.

    Args:
        model: Trained TorsionalDiffusionModel
        data: PyG Data object for the molecule
        mol: RDKit molecule object

    Returns:
        generated_torsions: Generated torsion angles [num_torsions]
        original_torsions: Original torsion angles [num_torsions]
    """
    model.eval()

    # Prepare input
    batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    num_torsions = len(data.torsion_angles)

    # Generate torsions
    generated_torsions = model.generate_torsions(
        node_features=data.x,
        pos=data.pos,
        edge_index=data.edge_index,
        torsion_to_atoms=data.torsion_to_atoms,
        batch=batch,
        num_torsions=num_torsions
    )

    return generated_torsions.squeeze().cpu(), data.torsion_angles.numpy()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TORSIONAL DIFFUSION MODEL WITH EGNN")
    print("Modern Implementation for Molecular Conformation Generation")
    print("=" * 70)

    # Example dataset with 8 molecules
    data_dict = {
        'smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C',      # Ibuprofen - not toxic
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine - not toxic
            'CC(=O)Oc1ccccc1C(=O)O',      # Aspirin - not toxic
            'CCO',                         # Ethanol - not toxic (at low doses)
            'c1ccccc1',                    # Benzene - toxic
            'C(Cl)(Cl)(Cl)Cl',            # Carbon tetrachloride - toxic
            'c1cc(ccc1N)N',                # p-Phenylenediamine - toxic
            'C1=CC=C(C=C1)O',             # Phenol - toxic
        ],
        'toxic': [0, 0, 0, 0, 1, 1, 1, 1]
    }

    print(f"\nDataset: {len(data_dict['smiles'])} molecules")
    print("\n" + "-" * 70)
    print("STEP 1: Processing molecules and extracting torsions")
    print("-" * 70)

    # Prepare dataset
    dataset = prepare_dataset(data_dict['smiles'])

    print(f"\n✓ Successfully processed {len(dataset)} molecules")

    # Print torsion statistics
    print("\n" + "-" * 70)
    print("TORSION STATISTICS")
    print("-" * 70)
    for data, mol, smiles in dataset:
        num_torsions = len(data.torsion_angles)
        num_atoms = mol.GetNumAtoms()
        print(f"• {smiles[:30]:30s} | Atoms: {num_atoms:3d} | Torsions: {num_torsions:2d}")

    # Train model
    print("\n" + "-" * 70)
    print("STEP 2: Training diffusion model")
    print("-" * 70)

    model = train_torsional_diffusion(
        dataset=dataset,
        atom_feature_dim=data.x.shape[-1],
        num_epochs=50,  # Reduced for demo
        lr=1e-4,
        hidden_dim=64,
        num_timesteps=100  # Reduced for faster demo
    )

    # Generate conformers
    print("\n" + "-" * 70)
    print("STEP 3: Generating new conformers")
    print("-" * 70)

    for i, (data, mol, smiles) in enumerate(dataset[:3]):  # Show first 3
        if len(data.torsion_angles) == 0:
            print(f"\n{i+1}. {smiles[:40]}")
            print("   No rotatable bonds - skipping")
            continue

        generated, original = generate_conformer(model, data, mol)

        print(f"\n{i+1}. {smiles[:40]}")
        print(f"   Original torsions (deg): {np.degrees(original)}")
        print(f"   Generated torsions (deg): {np.degrees(generated.numpy())}")
        print(f"   Difference (deg): {np.degrees(generated.numpy() - original)}")

    print("\n" + "=" * 70)
    print("SUMMARY: Torsional Diffusion with EGNN")
    print("=" * 70)
    print("""
    ✓ Successfully implemented modern torsion diffusion model

    Key Features:
    1. EGNN Backbone: SE(3)-equivariant message passing
       - Respects molecular symmetries (rotation, translation, reflection)
       - Processes 3D geometry directly

    2. Torsional Representation:
       - Only models flexible degrees of freedom
       - Much more efficient than full 3D diffusion
       - Respects chemical constraints (bond lengths, angles)

    3. Circular Diffusion:
       - Handles periodic nature of angles [-π, π]
       - Uses sin/cos encoding for continuity

    4. Modern Architecture:
       - Cosine noise schedule (better than linear)
       - Sinusoidal time embeddings
       - Residual connections in EGNN

    Applications:
    • Drug discovery: Generate diverse molecular conformations
    • Protein-ligand docking: Sample binding poses
    • Molecular dynamics: Explore conformational space

    Next Steps:
    • Train on larger dataset (1000s of molecules)
    • Add bond length/angle refinement
    • Implement conditional generation (target properties)
    • Add evaluation metrics (RMSD, energy)
    """)
    print("=" * 70)
