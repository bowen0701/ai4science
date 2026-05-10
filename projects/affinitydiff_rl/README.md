# AffinityDiff-RL: Diffusion Transformer with Offline Reinforcement Learning for High-affinity Protein-Ligan Generation

## ATOM3D Dataset Download

ATOM3D consists of two components:
1. **Python library** (`atom3d`) - Tools for reading LMDB format
2. **Dataset files** - The actual protein-ligand data (must download separately)

### Download Dataset Files

```bash
# Create data directory
mkdir -p projects/affinitydiff_rl/data
cd projects/affinitydiff_rl/data

# Download LBA dataset (2.3 GB)
wget https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz

# Extract
tar -xzf LBA-split-by-sequence-identity-30.tar.gz

# You should now have:
# projects/affinitydiff_rl/data/split-by-sequence-identity-30/data/
#   ├── train/data.mdb
#   ├── val/data.mdb
#   └── test/data.mdb
```

### Verify Installation

```python
import atom3d.datasets as da

# Load dataset (point to your extracted directory)
dataset = da.load_dataset('projects/affinitydiff_rl/data/split-by-sequence-identity-30/data/train', 'lmdb')

print(f"✓ Successfully loaded {len(dataset)} training examples")
# Expected: ✓ Successfully loaded ~3500 training examples
```

**Note**: The `atom3d` library does NOT automatically download datasets. You must manually download from Zenodo and then use the library to read the files.

**Notebook path note**: Paths above assume you run from the repo root. In notebooks under `projects/affinitydiff_rl/notebooks/`, use the relative path:
```python
dataset = da.load_dataset('../data/split-by-sequence-identity-30/data/train', 'lmdb')
```

### Basic Loading

```python
import atom3d.datasets as da

# Load dataset
dataset = da.load_dataset('projects/affinitydiff_rl/data/split-by-sequence-identity-30/data/train', 'lmdb')

# Access first entry
item = dataset[0]
print(f"PDB ID: {item['id']}")
print(f"Affinity: {item['scores']['neglog_aff']:.2f}")

# Get coordinates
protein_coords = item['atoms_protein'][['x', 'y', 'z']].values
ligand_coords = item['atoms_ligand'][['x', 'y', 'z']].values

print(f"Protein atoms: {protein_coords.shape}")
print(f"Ligand atoms: {ligand_coords.shape}")
```