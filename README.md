# NequIP-OpenMM Wrapper

## Problem

As of early 2026, the latest versions of NequIP (0.16.x) and OpenMM (8.x) have no direct interface. As NequIP no longer uses the `nequip-deploy` command to deploy MLPs, as their workflow shows (from https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html), but the NequIP interface for openmm-ml appears to still be written based on this order. 

<img width="1195" height="262" alt="image" src="https://github.com/user-attachments/assets/7354b01b-436d-4d56-8e93-85122eef6e5d" />


There is currently no official or third-party adapter available.

## Solution

`build_wrapper.py` bridges this gap by generating a TorchScript wrapper (`nequip_wrapped.pt`) that translates between the two frameworks:

- **Input**: OpenMM passes atomic positions as a single tensor in nanometers.
- **Output**: OpenMM expects either a scalar energy or an `(energy, forces)` tuple in kJ/mol and kJ/mol/nm.
- **NequIP expects**: A dictionary containing `pos`, `atom_types`, `edge_index`, and `edge_cell_shift`, with units in Angstroms and eV.

The wrapper handles unit conversion, constructs the NequIP input dictionary (including atom type mapping and all-pairs neighbor list), extracts the model output, and returns it in the format OpenMM requires. It also removes net force to prevent center-of-mass drift in gas-phase simulations.

## Requirements

- Python 3.8+
- PyTorch (compatible with your NequIP installation)
- NequIP 0.16.x
- ASE (for reading the PDB file)
- A deployed NequIP model (`lutein_deployed.nequip.pth`)
- The corresponding PDB structure file (`lutein.pdb`)

## Usage

Place `build_wrapper.py`, `lutein_deployed.nequip.pth`, and `lutein.pdb` in the same directory, then run:

```bash
python build_wrapper.py
```

This produces `nequip_wrapped.pt`, which can be loaded directly by OpenMM's TorchForce:

```python
from openmmtorch import TorchForce

torch_force = TorchForce('nequip_wrapped.pt')
torch_force.setOutputsForces(True)
system.addForce(torch_force)
```

## Important Notes

- The atom type mapping is hardcoded to match the NequIP training configuration (`model_type_names: [C, H, O]`, i.e. C=0, H=1, O=2). If your model was trained with a different ordering, you must update the `Z_to_type` dictionary in `build_wrapper.py`.
- The wrapper uses an all-pairs neighbor list, which is suitable for small isolated molecules (< ~200 atoms) but does not scale to large systems. For protein-scale simulations, a dynamic cutoff-based neighbor list would be needed.
- For gas-phase (vacuum) simulations, always add `openmm.CMMotionRemover(1)` to the system to prevent center-of-mass drift.
