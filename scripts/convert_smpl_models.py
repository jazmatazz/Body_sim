#!/usr/bin/env python3
"""
Convert SMPL model pickle files to remove chumpy dependencies.

SMPL pickle files contain chumpy arrays which require the chumpy library.
This script converts them to pure numpy arrays for compatibility with
modern Python versions where chumpy doesn't install.

Usage:
    python scripts/convert_smpl_models.py

This will create converted versions with _numpy suffix.
"""

import pickle
import sys
from pathlib import Path
import numpy as np


def convert_chumpy_to_numpy(obj):
    """Recursively convert chumpy arrays to numpy arrays."""
    if isinstance(obj, dict):
        return {k: convert_chumpy_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_chumpy_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_chumpy_to_numpy(v) for v in obj)
    elif isinstance(obj, FakeChumpy):
        # Our fake chumpy object
        return np.array(obj.r)
    elif hasattr(obj, 'r') and not isinstance(obj, np.ndarray):
        # Chumpy array - has .r attribute for the numpy array
        try:
            return np.array(obj.r)
        except:
            return obj
    elif hasattr(obj, '__array__') and not isinstance(obj, np.ndarray):
        try:
            return np.array(obj)
        except:
            return obj
    else:
        return obj


class FakeModule:
    """Fake module to handle chumpy imports during unpickling."""
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        return FakeChumpy


class FakeChumpy:
    """Fake chumpy array that stores data for later conversion."""
    def __init__(self, *args, **kwargs):
        if args:
            self._data = np.asarray(args[0]) if args[0] is not None else np.array([])
        else:
            self._data = np.array([])

    def __setstate__(self, state):
        """Handle unpickling - state is usually the numpy array data."""
        if isinstance(state, dict):
            self._data = state.get('x', state.get('_data', np.array([])))
            if hasattr(self._data, 'r'):
                self._data = np.asarray(self._data.r)
        elif isinstance(state, np.ndarray):
            self._data = state
        elif hasattr(state, '__array__'):
            self._data = np.asarray(state)
        else:
            self._data = np.array([])

    def __reduce__(self):
        return (np.array, (self._data,))

    @property
    def r(self):
        if hasattr(self, '_data'):
            return self._data
        return np.array([])


class ChumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that handles chumpy references."""

    def find_class(self, module, name):
        # Handle chumpy references
        if 'chumpy' in module:
            return FakeChumpy
        # Handle scipy.sparse references
        if module == 'scipy.sparse.csc':
            import scipy.sparse
            return getattr(scipy.sparse, name)
        if module == 'scipy.sparse.csr':
            import scipy.sparse
            return getattr(scipy.sparse, name)
        return super().find_class(module, name)


def load_smpl_model(filepath: Path) -> dict:
    """Load SMPL model handling chumpy arrays."""
    with open(filepath, 'rb') as f:
        try:
            # Try custom unpickler first
            unpickler = ChumpyUnpickler(f, encoding='latin1')
            data = unpickler.load()
        except Exception as e:
            print(f"  Custom unpickler failed: {e}")
            print("  Trying alternative method...")
            f.seek(0)
            # Alternative: try direct pickle with module override
            import sys

            # Create fake chumpy module
            class FakeCh:
                class Ch:
                    def __init__(self, *args, **kwargs):
                        self.x = args[0] if args else np.array([])
                    @property
                    def r(self):
                        return np.asarray(self.x)
                    def __array__(self):
                        return np.asarray(self.x)

                array = Ch

            sys.modules['chumpy'] = FakeCh()
            sys.modules['chumpy.ch'] = FakeCh()

            try:
                data = pickle.load(f, encoding='latin1')
            finally:
                # Clean up
                if 'chumpy' in sys.modules:
                    del sys.modules['chumpy']
                if 'chumpy.ch' in sys.modules:
                    del sys.modules['chumpy.ch']

    return data


def convert_model(input_path: Path, output_path: Path) -> bool:
    """Convert a single SMPL model file."""
    print(f"  Loading {input_path.name}...")

    try:
        data = load_smpl_model(input_path)
    except Exception as e:
        print(f"  Error loading: {e}")
        return False

    print(f"  Converting chumpy arrays to numpy...")
    converted = convert_chumpy_to_numpy(data)

    # Verify key fields exist
    required_keys = ['v_template', 'shapedirs', 'J_regressor', 'weights', 'posedirs', 'f']
    for key in required_keys:
        if key not in converted:
            print(f"  Warning: missing key '{key}'")

    print(f"  Saving to {output_path.name}...")
    with open(output_path, 'wb') as f:
        pickle.dump(converted, f, protocol=2)

    return True


def main():
    model_dir = Path(__file__).parent.parent / "models" / "smpl"

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1

    print("=" * 60)
    print("SMPL Model Converter - Remove chumpy dependencies")
    print("=" * 60)
    print(f"\nModel directory: {model_dir}")

    # Find model files to convert
    model_files = [
        "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
        "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
        "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    ]

    converted_count = 0

    for filename in model_files:
        input_path = model_dir / filename
        if not input_path.exists():
            print(f"\nSkipping {filename} (not found)")
            continue

        # Output to smpl subdirectory with standard names
        smpl_dir = model_dir / "smpl"
        smpl_dir.mkdir(exist_ok=True)

        # Map to standard names
        if '_m_' in filename:
            output_name = "SMPL_MALE.pkl"
        elif '_f_' in filename:
            output_name = "SMPL_FEMALE.pkl"
        else:
            output_name = "SMPL_NEUTRAL.pkl"

        output_path = smpl_dir / output_name

        # Remove existing symlink if present
        if output_path.is_symlink():
            output_path.unlink()

        print(f"\nConverting {filename} -> smpl/{output_name}")

        if convert_model(input_path, output_path):
            converted_count += 1
            print(f"  ✓ Success!")
        else:
            print(f"  ✗ Failed!")

    print("\n" + "=" * 60)
    print(f"Converted {converted_count}/{len(model_files)} models")

    if converted_count > 0:
        print("\nConverted models saved to: models/smpl/smpl/")
        print("You can now run simulations without chumpy!")

    return 0 if converted_count == len(model_files) else 1


if __name__ == "__main__":
    sys.exit(main())
