"""
Run this ONCE to generate 'nequip_wrapped.pt'.

Key fixes:
  - Remove net force (zero COM force) to prevent molecular drift
  - Proper detach/requires_grad handling for NequIP internal autograd

Usage:
    python build_wrapper.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from ase.io import read

# ============================================================
# Read atom types from PDB
# ============================================================
atoms = read('lutein.pdb')
atomic_numbers = atoms.get_atomic_numbers()

# Build atom_types mapping
# MUST match training config: model_type_names: [C, H, O]
# So: C=0, H=1, O=2
type_name_to_index = {'C': 0, 'H': 1, 'O': 2}
Z_to_type = {6: 0, 1: 1, 8: 2}  # C(Z=6)->0, H(Z=1)->1, O(Z=8)->2
atom_types = [Z_to_type[z] for z in atomic_numbers]

print(f"Type mapping (from training config): C=0, H=1, O=2")
print(f"Z -> type: {Z_to_type}")
print(f"Number of atoms: {len(atom_types)}")
print(f"Type counts: C={atom_types.count(0)}, H={atom_types.count(1)}, O={atom_types.count(2)}")

# Get masses for COM force removal
masses = atoms.get_masses()
print(f"Total mass: {masses.sum():.2f} amu")

# ============================================================
# Load NequIP compiled model
# ============================================================
nequip_model = torch.jit.load('lutein_deployed.nequip.pth', map_location='cpu')

# ============================================================
# Build all-pairs neighbor list (for small molecule, no PBC)
# ============================================================
n_atoms = len(atom_types)

edge_index_list = []
for i in range(n_atoms):
    for j in range(n_atoms):
        if i != j:
            edge_index_list.append([i, j])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
print(f"Number of edges (all pairs): {edge_index.shape[1]}")


# ============================================================
# Wrapper Module
# ============================================================
class NequIPWrapper(nn.Module):
    """Wraps a NequIP compiled model for use with OpenMM TorchForce.
    
    Returns (energy, forces) tuple for setOutputsForces(True).
    Forces have net force removed to prevent COM drift.
    """
    def __init__(self, nequip_model, atom_types, edge_index, masses):
        super().__init__()
        self.model = nequip_model
        self.register_buffer('atom_types', torch.tensor(atom_types, dtype=torch.long))
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_cell_shift', torch.zeros((edge_index.shape[1], 3), dtype=torch.float64))
        
        # Masses for COM force removal (n_atoms, 1)
        mass_tensor = torch.tensor(masses, dtype=torch.float64).unsqueeze(1)
        self.register_buffer('masses', mass_tensor)
        self.register_buffer('total_mass', mass_tensor.sum())
        
        # Unit conversion constants
        self.nm_to_angstrom = 10.0
        self.eV_to_kJ_per_mol = 96.4853
        self.force_conversion = 96.4853 * 10.0  # eV/Å -> kJ/mol/nm

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: (n_atoms, 3) in nm from OpenMM
        pos = positions.to(torch.float64) * self.nm_to_angstrom
        pos = pos.detach().requires_grad_(True)
        
        # Build input dict
        data: Dict[str, torch.Tensor] = {
            'pos': pos,
            'atom_types': self.atom_types,
            'edge_index': self.edge_index,
            'edge_cell_shift': self.edge_cell_shift,
        }
        
        # Run NequIP model
        result = self.model(data)
        
        # Energy (eV -> kJ/mol)
        energy = result['total_energy'].to(torch.float64).squeeze() * self.eV_to_kJ_per_mol
        
        # Forces (eV/Å -> kJ/mol/nm)
        forces = result['forces'].to(torch.float64) * self.force_conversion
        
        # Remove net force to prevent COM drift
        # Mass-weighted: subtract (m_i / M_total) * F_net from each atom
        net_force = forces.sum(dim=0, keepdim=True)  # (1, 3)
        forces = forces - (self.masses / self.total_mass) * net_force
        
        return energy, forces


# ============================================================
# Create and save wrapper
# ============================================================
wrapper = NequIPWrapper(nequip_model, atom_types, edge_index, masses)
wrapper.eval()

# Test with initial positions
pos_test = torch.tensor(atoms.get_positions(), dtype=torch.float64) * 0.1  # Å -> nm
energy_test, forces_test = wrapper(pos_test)
print(f"Test energy: {energy_test.item():.4f} kJ/mol")
print(f"Test forces shape: {forces_test.shape}")
print(f"Test forces max abs: {forces_test.abs().max().item():.4f} kJ/mol/nm")
print(f"Net force after removal: {forces_test.sum(dim=0).abs().max().item():.2e} kJ/mol/nm")

# Script and save
scripted = torch.jit.script(wrapper)
scripted.save('nequip_wrapped.pt')
print("Saved wrapper to nequip_wrapped.pt")
