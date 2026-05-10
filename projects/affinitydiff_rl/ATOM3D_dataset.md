---
title: "ATOM3D Dataset"
created: 2026-05-10
tags:
  - dataset
  - protein-ligand
  - binding-affinity
  - 3d-ml
  - atom3d
  - pdbbind
  - drug-discovery
  - structural-biology
  - sequence-identity
  - data-leakage-prevention
---

# ATOM3D: Ligand Binding Affinity (LBA) Dataset

## Overview

**Task**: Predict protein-ligand binding affinity from 3D crystal structures.  
**Source**: PDBbind v2016 refined set  
**Format**: LMDB  
**Paper**: "ATOM3D: Tasks On Molecules in Three Dimensions" (NeurIPS 2021)  
**Homepage**: https://www.atom3d.ai/  
**Download**: https://zenodo.org/record/4914718

| Split | Complexes | Split Method |
|-------|-----------|--------------|
| Train | ~3,500 | 30% sequence identity |
| Val | ~460 | 30% sequence identity |
| Test | ~460 | 30% sequence identity |

**Affinity range**: 2.0 – 12.0 (-log Kd/Ki)  
**Avg ligand size**: ~25 heavy atoms  
**Total size**: 2.3 GB

---

## Task Framing

Each record is one PDB crystal structure: a `(protein, ligand) → affinity` triple.

```python
{
    'id': str,                  # PDB ID, e.g. '1ugx'
    'atoms_protein': DataFrame, # all protein heavy atoms with 3D coords
    'atoms_ligand': DataFrame,  # all ligand heavy atoms with 3D coords
    'scores': {
        'neglog_aff': float     # -log(Kd/Ki), higher = stronger binding
    }
}
```

The target is a **regression** on `scores['neglog_aff']`.

---

## Protein & Ligand Atom Structure

Both `atoms_protein` and `atoms_ligand` are flat DataFrames at **atom-level** granularity — not residue-level.

### Protein

A protein is a chain of amino acid residues; each residue is made up of heavy atoms. `atoms_protein` flattens all of them:

```
protein
└── residues (resname: GLY, ALA, PHE, ...)
    └── atoms per residue
        ├── N    — backbone nitrogen        (every residue)
        ├── CA   — alpha carbon             (every residue)
        ├── C    — backbone carbonyl carbon (every residue)
        ├── O    — backbone oxygen          (every residue)
        └── CB, CG, OD1, ... — side-chain atoms (residue-specific)
```

`N`, `CA`, `C`, `O` are the backbone atoms shared by all residues. Side-chain atoms vary by residue type and encode the chemical identity of each amino acid. Water molecules (`HOH`, `hetero == 'W'`) are also included and need to be masked out.

### Ligand

`atoms_ligand` contains the small molecule heavy atoms (`C1`, `C2`, `O1`, ...) with their crystal-structure 3D positions. ATOM3D renames all ligands to the generic `resname = "LIG"`, losing the original 3-letter PDB code.

### DataFrame Columns (shared schema)

| Column | Description |
|--------|-------------|
| `element` | Atom element symbol (C, N, O, S, ...) |
| `x`, `y`, `z` | 3D coordinates in Angstroms |
| `resname` | Residue/ligand name (e.g. GLY, LIG) |
| `residue` | Residue index |
| `name` / `fullname` | Atom name (e.g. CA, CB, C1) |
| `chain` | Chain identifier |
| `hetero` | `'W'` for water, empty for protein atoms |
| `bfactor` | B-factor (crystallographic temperature factor, proxy for flexibility) |
| `occupancy` | Occupancy (usually 1.0) |

Models operate either at the atom level (equivariant GNNs) or aggregate atoms → residues first (residue-level graph networks).

---

## Sequence Identity Split

**The most important modeling decision in this dataset.**

Sequence identity = % of identical amino acids between two protein sequences. Any two proteins across train/val/test have **< 30% sequence identity**.

| Split method | Typical Pearson R | What it tests |
|---|---|---|
| Random split | 0.82 | Memorization of protein families |
| **30% identity split** | **0.67** | **Generalization to novel protein families** |

Random splits leak: proteins with > 70% identity share nearly identical 3D structures and binding pockets, so the model just memorizes family-level patterns. The 30% cutoff tests the practically meaningful question for drug discovery: *can your model predict binding for completely novel protein targets?*

---

## Model Input Featurization

### Record Keys → Model Inputs

**Protein** (`atoms_protein`)

| Feature | Key | Transformation |
|---|---|---|
| 3D coordinates | `x`, `y`, `z` | stack → `(N, 3)` float tensor |
| Atom element | `element` | one-hot `{C, N, O, S, H, ...}` |
| Residue type | `resname` | one-hot 20 amino acids |
| Atom role | `name` | one-hot `{CA, CB, N, C, O}` or embed |
| Flexibility | `bfactor` | normalize → scalar node feature |
| Water mask | `hetero == 'W'` | binary flag to exclude waters |
| Grouping | `residue`, `chain` | aggregate atom → residue representations |

**Ligand** (`atoms_ligand`)

| Feature | Key | Transformation |
|---|---|---|
| 3D coordinates | `x`, `y`, `z` | stack → `(M, 3)` float tensor |
| Atom element | `element` | one-hot `{C, N, O, S, ...}` |
| Bond type | not in record | need RDKit from SMILES → single/double/aromatic |
| Hybridization | not in record | need RDKit → sp/sp2/sp3 |
| Aromaticity | not in record | need RDKit → binary flag |
| H-bond donor/acceptor | not in record | need RDKit → binary flags |

**Interaction** (derived)

| Feature | Source | Transformation |
|---|---|---|
| Pairwise distances | protein + ligand `x, y, z` | `cdist` → `(N, M)` |
| Contact map | derived | binary mask where distance `< 5Å` |
| Pocket atoms | derived | protein atoms within `10Å` of ligand centroid |

The 5Å cutoff captures all direct non-covalent interactions that drive binding:

| Interaction type | Typical distance |
|---|---|
| Hydrogen bonds | 2.5 – 3.5 Å |
| Van der Waals contacts | 3.0 – 4.0 Å |
| Hydrophobic contacts | 3.5 – 5.0 Å |
| Ionic / electrostatic | up to ~5 Å |

Beyond 5Å, atoms are too far apart to interact directly. The contact map gives a sparse binary mask of which protein-ligand atom pairs are in contact — defining interaction edges in a GNN.

The 10Å pocket cutoff works by computing each protein atom's distance to the ligand centroid:

```python
ligand_centroid = ligand_coords.mean(axis=0)        # (3,)  — geometric center of ligand

# Broadcasting: subtract one point from every protein atom row
# protein_coords: (9512, 3) - ligand_centroid: (3,) → (9512, 3) displacements
pocket_dists = np.linalg.norm(protein_coords - ligand_centroid, axis=1)
# axis=1: sqrt(dx² + dy² + dz²) per row → shape (9512,) one distance per atom

pocket_mask = pocket_dists < pocket_cutoff          # (9512,) bool mask
pocket_coords = protein_coords[pocket_mask]         # (~200–500, 3)
```

Visually:
```
ligand centroid ●
                │← 10Å →│
    ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ○   ← protein atoms
    ✗ ✗ ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✗ ✗ ✗   ← pocket_mask
```

The centroid is a coarser anchor than per-atom distances (used by the 5Å contact map), but fast and sufficient for defining the pocket boundary.

Those ~200 pocket atoms are the only ones that matter for affinity — they are the **binding site**, the only protein atoms close enough to the ligand to influence binding. The other ~9,300 atoms are on the surface, buried in the protein core, or on the opposite side — too far away to interact with the ligand. Binding affinity is determined entirely by local pocket geometry and chemistry, not the rest of the protein.

For `1ugx` concretely:
```
full protein:   9,512 atoms  (~1,200 residues — entire chain)
water removed:  ~9,000 atoms
pocket (10Å):     ~200 atoms (~25 residues — binding site)
contacts (5Å):     ~50 atoms (~6 residues  — directly touching ligand)
ligand:            27 atoms
```

Every published 3D affinity model (EquiBind, DiffDock, EGNN-based scorers) uses pocket-only input. Using the full protein is also impractical: 9,500-atom graphs × batch size in a GNN means enormous graphs with mostly noise edges.

The two cutoffs work together:

```
all protein atoms (~9,500)
    → 10Å from ligand centroid → pocket atoms (~200)
        → 5Å from any ligand atom → direct contacts (GNN edges)
```

**Target**

| Feature | Key | Notes |
|---|---|---|
| Binding affinity | `scores['neglog_aff']` | `-log(Kd)`, direct regression target |

### Full Chain for GNN Featurization

Bond-level features (hybridization, aromaticity, bond type) are not in the record — they require RDKit from SMILES. Since ATOM3D renames all ligands to `"LIG"`, the original 3-letter PDB code must be recovered via the RCSB API.

```
item['id'] (PDB ID, e.g. '1ugx')
    → RCSB API → 3-letter comp_id (real ligand code, e.g. 'ATP')
    → CCD → SMILES string
    → RDKit mol → bond graph + hybridization + aromaticity
    + item['atoms_ligand'][x, y, z, element] → 3D positions
    → featurized molecular graph for GNN
```

**Step 1 — Get SMILES via RCSB REST API**

```python
import requests

def get_ligand_smiles(pdb_id: str) -> dict[str, str]:
    """Returns {comp_id: SMILES} for all ligands in a PDB entry."""
    base = "https://data.rcsb.org/rest/v1/core"

    entry = requests.get(f"{base}/entry/{pdb_id}").json()
    entity_ids = (entry
        .get("rcsb_entry_container_identifiers", {})
        .get("non_polymer_entity_ids", []))

    smiles = {}
    for eid in entity_ids:
        entity = requests.get(f"{base}/nonpolymer_entity/{pdb_id}/{eid}").json()
        comp_id = entity.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if not comp_id:
            continue
        ccd = requests.get(f"{base}/chemcomp/{comp_id}").json()
        for entry in ccd.get("pdbx_chem_comp_identifier", []):
            if "SMILES" in entry.get("type", ""):
                smiles[comp_id] = entry["identifier"]
                break
    return smiles

smiles_map = get_ligand_smiles(item['id'])
```

**Step 2 — Build molecular graph with RDKit**

```python
from rdkit import Chem

comp_id, smi = next(iter(smiles_map.items()))
mol = Chem.MolFromSmiles(smi)

for atom in mol.GetAtoms():
    print(atom.GetSymbol(), atom.GetHybridization(),
          atom.GetIsAromatic(), atom.GetFormalCharge(), atom.GetTotalNumHs())

for bond in mol.GetBonds():
    print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
          bond.GetBondTypeAsDouble(), bond.GetIsAromatic(), bond.IsInRing())
```

**Step 3 — Combine with ATOM3D 3D positions**

```python
import numpy as np

ligand_coords = item['atoms_ligand'][['x', 'y', 'z']].values  # (M, 3)

node_features = []
for atom in mol.GetAtoms():
    node_features.append([
        *ligand_coords[atom.GetIdx()],
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()),
        int(atom.GetHybridization()),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
    ])

edge_index = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
```

> If a PDB entry has multiple ligands, match the right `comp_id` by checking which component has the same heavy atom count as `len(atoms_ligand)`.

---

## Setup & Loading

```bash
mkdir -p projects/affinitydiff_rl/data
cd projects/affinitydiff_rl/data
wget https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz
tar -xzf LBA-split-by-sequence-identity-30.tar.gz
# → split-by-sequence-identity-30/data/{train,val,test}/data.mdb
```

```python
import atom3d.datasets as da
from pathlib import Path

data_path = Path.cwd().parent / 'data/split-by-sequence-identity-30/data/train'
dataset = da.load_dataset(str(data_path), 'lmdb')
print(f"✓ Loaded {len(dataset)} training examples")

item = dataset[0]
print(item['id'], item['scores']['neglog_aff'])
```

---

## PyTorch Integration

### Dataset Wrapper

`atoms_protein` is the full protein (~9,500 atoms). Pocket extraction (10Å from ligand centroid) is a must-have step that reduces it to the ~200 binding-relevant atoms before passing to any model.

```python
import numpy as np
import torch
from torch.utils.data import Dataset
import atom3d.datasets as da

POCKET_CUTOFF = 10.0  # Å from ligand centroid → binding pocket
ELEMENT_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
RESIDUE_LIST = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL',
]

class LBADataset(Dataset):
    def __init__(self, lmdb_path):
        self.dataset = da.load_dataset(lmdb_path, 'lmdb')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- Ligand ---
        # coords: WHERE each atom is; feats: WHAT each atom is (element type)
        lig = item['atoms_ligand']
        ligand_coords = lig[['x', 'y', 'z']].values                    # (M, 3)
        ligand_feats = self._element_onehot(lig['element'])             # (M, 9)

        # --- Pocket extraction ---
        # atoms_protein is the full protein (~9,500 atoms); extract binding pocket
        prot = item['atoms_protein']
        prot = prot[prot['hetero'] != 'W']                              # remove water
        protein_coords = prot[['x', 'y', 'z']].values                  # (N, 3)

        ligand_centroid = ligand_coords.mean(axis=0)                    # (3,)
        pocket_dists = np.linalg.norm(protein_coords - ligand_centroid, axis=1)
        pocket_mask = pocket_dists < POCKET_CUTOFF                      # (N,) bool

        pocket_prot = prot[pocket_mask]
        pocket_coords = protein_coords[pocket_mask]                     # (~200, 3)

        # coords: WHERE; feats: WHAT — element (9) + residue/amino acid type (20) = 29
        pocket_feats = np.concatenate([
            self._element_onehot(pocket_prot['element']),               # (~200, 9)
            self._residue_onehot(pocket_prot['resname']),               # (~200, 20)
        ], axis=1)                                                       # (~200, 29)

        # --- Target ---
        affinity = torch.FloatTensor([item['scores']['neglog_aff']])

        return {
            'id': item['id'],
            'pocket_coords': torch.FloatTensor(pocket_coords),
            'pocket_feats': torch.FloatTensor(pocket_feats),
            'ligand_coords': torch.FloatTensor(ligand_coords),
            'ligand_feats': torch.FloatTensor(ligand_feats),
            'affinity': affinity,
        }

    def _element_onehot(self, elements) -> np.ndarray:
        feats = np.zeros((len(elements), len(ELEMENT_LIST)), dtype=np.float32)
        for i, elem in enumerate(elements):
            if elem in ELEMENT_LIST:
                feats[i, ELEMENT_LIST.index(elem)] = 1.0
        return feats

    def _residue_onehot(self, resnames) -> np.ndarray:
        feats = np.zeros((len(resnames), len(RESIDUE_LIST)), dtype=np.float32)
        for i, res in enumerate(resnames):
            if res in RESIDUE_LIST:
                feats[i, RESIDUE_LIST.index(res)] = 1.0
        return feats
```

### Tensor Shapes

3D coordinates tell the model **where** each atom is; features tell it **what** each atom is. Both are required — coordinates alone are anonymous 3D points the model can't interpret chemically.

| Tensor | Shape | Meaning |
|---|---|---|
| `pocket_coords` | `(~200, 3)` | WHERE: ~200 pocket atoms × `(x, y, z)` |
| `pocket_feats` | `(~200, 29)` | WHAT: element one-hot (9) + amino acid one-hot (20) |
| `ligand_coords` | `(M, 3)` | WHERE: M ligand atoms × `(x, y, z)` |
| `ligand_feats` | `(M, 9)` | WHAT: element one-hot (9) |

Element one-hot encodes atom type; `resname` one-hot encodes which of the 20 amino acids the atom belongs to — capturing side-chain chemistry that element alone misses (e.g. `C` in `PHE` is aromatic; `C` in `GLY` is a backbone carbon):

```
atom element = 'N'  →  [0, 1, 0, 0, 0, 0, 0, 0, 0]   (9-dim)
                         C  N  O  S  F  P  Cl Br  I

atom resname = 'PHE' →  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   (20-dim)
                         A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
```

Coordinates and features are kept as separate tensors because they play different roles in equivariant models: coordinates transform under rotation/translation, features do not.

### Common Pitfalls

**Skipping pocket extraction** — `atoms_protein` is the full protein (~9,500 atoms), not the pocket. Without the 10Å cutoff you feed 47× more atoms than needed into the model:
```python
# ❌ full protein — ~9,500 atoms, most irrelevant to binding
protein_coords = item['atoms_protein'][['x', 'y', 'z']].values

# ✅ binding pocket — ~200 atoms within 10Å of ligand centroid
ligand_centroid = ligand_coords.mean(axis=0)
pocket_mask = np.linalg.norm(protein_coords - ligand_centroid, axis=1) < 10.0
pocket_coords = protein_coords[pocket_mask]
```

**Forgetting to remove water** — `atoms_protein` includes `HOH` water molecules (`hetero == 'W'`). Filter them before pocket extraction:
```python
prot = item['atoms_protein']
prot = prot[prot['hetero'] != 'W']  # remove water before any featurization
```

**Variable-size batching** — pocket and ligand atom counts differ per complex; don't stack directly:
```python
def collate_fn(batch):
    return {
        'pocket_coords': [x['pocket_coords'] for x in batch],
        'pocket_feats': [x['pocket_feats'] for x in batch],
        'ligand_coords': [x['ligand_coords'] for x in batch],
        'ligand_feats': [x['ligand_feats'] for x in batch],
        'affinity': torch.stack([x['affinity'] for x in batch]),
    }
```

**Memory** — LMDB is memory-mapped; never load the full dataset into RAM:
```python
dataset = da.load_dataset(path, 'lmdb')                       # ✅ lazy
all_data = [dataset[i] for i in range(len(dataset))]          # ❌ 2 GB into RAM
```

**Coordinate normalization** — each complex has a different center-of-mass; center on the ligand or use SE(3)-equivariant models:
```python
ligand_center = ligand_coords.mean(axis=0)
pocket_coords = pocket_coords - ligand_center
ligand_coords = ligand_coords - ligand_center
```

---

## Storage Format (LMDB)

LMDB (Lightning Memory-Mapped Database) is a key-value store where each value is a pickled Python dict. It's memory-mapped — the OS pages in only what you access, so the 2 GB file never fully loads into RAM.

```
LBA-split-by-sequence-identity-30/
└── data/
    ├── train/ {data.mdb (~2.0 GB), lock.mdb (8 KB)}
    ├── val/   {data.mdb (~250 MB), lock.mdb}
    └── test/  {data.mdb (~250 MB), lock.mdb}
```

`lock.mdb` is a write-lock file — irrelevant for read-only access.

| Format | Read speed | Memory | Random access |
|---|---|---|---|
| Individual PDB files | Slow (parse each time) | Low | Slow |
| Pickle / HDF5 | Medium | High (must load) | Medium |
| **LMDB** | **Fast** | **Low (memory-mapped)** | **Fast O(1)** |

All metadata (PDB IDs, affinities, splits) is embedded in the LMDB — no separate CSV or JSON files to keep in sync.

---

## Benchmark Results

On the 30% sequence identity split:

| Model | Test RMSE | Test Pearson R |
|-------|-----------|----------------|
| 3D CNN | 1.54 | 0.61 |
| GCN | 1.48 | 0.64 |
| EGNN | 1.42 | 0.67 |
| Equiformer | 1.38 | 0.72 |
| TorchMD-NET | 1.35 | 0.74 |

These reflect true generalization — not inflated random-split metrics.

---

## Comparison with Other Datasets

| Dataset | Size | Split | Affinity labels | Use case |
|---|---|---|---|---|
| **ATOM3D LBA** | 4.4k | Sequence identity | Yes | Rigorous 3D ML benchmark |
| PDBbind Refined | 5.3k | Random/temporal | Yes | General benchmarking |
| PDBbind Core | 285 | Curated test set | Yes | Final evaluation |
| Binding MOAD | 40k | Various | Yes | Large-scale training |
| CrossDocked | 100k+ | Docking poses | No | Docking validation |

**Pros**: ready to use, rigorous split, standard benchmark, efficient LMDB format.  
**Cons**: small (4k vs 19k in PDBbind general), 2016 vintage, heavy atoms only (no H).

---

## Citation

```bibtex
@inproceedings{townshend2021atom3d,
  title={ATOM3D: Tasks on molecules in three dimensions},
  author={Townshend, Raphael JL and others},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2021}
}
```

---

## Related Notes

- [[PDBBind Dataset]]
- TODO: [[SE(3) Equivariant Neural Networks]]
- TODO: [[Protein-Ligand Docking]]
- TODO: [[Diffusion Models for Molecules]]
- TODO: [[PyTorch Geometric for 3D Data]]
