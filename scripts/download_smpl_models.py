#!/usr/bin/env python3
"""
Script to guide users through SMPL model setup.

SMPL models cannot be automatically downloaded due to licensing.
Users must register at https://smpl.is.tue.mpg.de/ and download manually.

This script:
1. Checks if models are already present
2. Provides instructions for obtaining models
3. Validates model files after download
"""

import sys
from pathlib import Path


# Default model directory
MODEL_DIR = Path(__file__).parent.parent / "models" / "smpl"

# Required model files for smplpytorch
REQUIRED_FILES = {
    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl": "Male model",
    "basicmodel_f_lbs_10_207_0_v1.1.0.pkl": "Female model",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": "Neutral model",
}

# Alternative filenames from different SMPL versions
ALTERNATIVE_FILES = {
    "SMPL_MALE.pkl": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
    "SMPL_FEMALE.pkl": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
    "SMPL_NEUTRAL.pkl": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
}


def check_models(model_dir: Path = MODEL_DIR) -> tuple[bool, list[str]]:
    """Check if SMPL models are present.

    Args:
        model_dir: Directory to check for models

    Returns:
        (all_present, missing_files) tuple
    """
    model_dir = Path(model_dir)
    missing = []

    for filename, description in REQUIRED_FILES.items():
        filepath = model_dir / filename

        # Check for alternative names
        found = filepath.exists()
        if not found:
            for alt_name, std_name in ALTERNATIVE_FILES.items():
                if std_name == filename and (model_dir / alt_name).exists():
                    found = True
                    print(f"  Found alternative: {alt_name} -> {filename}")
                    break

        if not found:
            missing.append(filename)

    return len(missing) == 0, missing


def print_instructions():
    """Print instructions for obtaining SMPL models."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    SMPL Model Download Instructions                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  SMPL models are required for body mesh generation but cannot be    ║
║  automatically downloaded due to licensing requirements.             ║
║                                                                      ║
║  To obtain SMPL models:                                             ║
║                                                                      ║
║  1. Visit: https://smpl.is.tue.mpg.de/                              ║
║                                                                      ║
║  2. Register for a free account (academic/research use)             ║
║                                                                      ║
║  3. Download "SMPL for Python" package                              ║
║                                                                      ║
║  4. Extract the model files to:                                     ║
║     {model_dir}
║                                                                      ║
║  Required files:                                                     ║
║  - basicmodel_m_lbs_10_207_0_v1.1.0.pkl (male)                      ║
║  - basicmodel_f_lbs_10_207_0_v1.1.0.pkl (female)                    ║
║  - basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl (neutral)             ║
║                                                                      ║
║  After downloading, run this script again to verify installation.   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(model_dir=MODEL_DIR.absolute()))


def validate_model_file(filepath: Path) -> bool:
    """Validate that a model file is a valid SMPL pickle.

    Args:
        filepath: Path to model file

    Returns:
        True if valid SMPL model
    """
    import pickle
    import struct

    if not filepath.exists():
        return False

    # Check file size (SMPL models are typically 20-50MB)
    file_size = filepath.stat().st_size
    if file_size < 1_000_000:  # < 1MB is too small
        print(f"  Warning: {filepath.name} is too small ({file_size} bytes)")
        return False

    # Check it's a valid pickle file by reading the header
    try:
        with open(filepath, "rb") as f:
            # Read first few bytes to check pickle signature
            header = f.read(2)
            # Pickle files typically start with specific opcodes
            # \x80 is PROTO opcode, common in pickle protocol 2+
            if header[0:1] not in (b'\x80', b'(', b'}', b']'):
                print(f"  Warning: {filepath.name} doesn't appear to be a pickle file")
                return False

            # Try to scan for expected SMPL keys in the file
            f.seek(0)
            content = f.read()
            expected_keys = [b'v_template', b'shapedirs', b'J_regressor', b'weights']
            found_keys = sum(1 for key in expected_keys if key in content)

            if found_keys < 3:
                print(f"  Warning: {filepath.name} missing expected SMPL data structures")
                return False

        return True

    except Exception as e:
        print(f"  Error validating {filepath.name}: {e}")
        return False


def create_model_directory():
    """Create the model directory if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created model directory: {MODEL_DIR.absolute()}")


def main():
    print("=" * 70)
    print("SMPL Model Setup for Body Pressure Simulator")
    print("=" * 70)

    # Create directory if needed
    if not MODEL_DIR.exists():
        create_model_directory()

    print(f"\nChecking for models in: {MODEL_DIR.absolute()}")
    print()

    # Check for models
    all_present, missing = check_models()

    if all_present:
        print("✓ All required SMPL models found!")
        print()

        # Validate models
        print("Validating model files...")
        all_valid = True
        for filename in REQUIRED_FILES.keys():
            filepath = MODEL_DIR / filename
            if filepath.exists():
                if validate_model_file(filepath):
                    print(f"  ✓ {filename}: Valid")
                else:
                    print(f"  ✗ {filename}: Invalid or corrupted")
                    all_valid = False

        if all_valid:
            print()
            print("=" * 70)
            print("✓ SMPL models are properly installed and validated!")
            print("  You can now run simulations with real body models.")
            print("=" * 70)
            return 0
        else:
            print()
            print("Some models failed validation. Please re-download them.")
            return 1

    else:
        print("✗ Missing SMPL model files:")
        for filename in missing:
            print(f"  - {filename} ({REQUIRED_FILES[filename]})")

        print_instructions()

        # Check if user wants to use mock model instead
        print("Alternative: You can run simulations without SMPL models")
        print("by using the --mock flag with the simulator.")
        print()
        print("Example:")
        print("  python examples/basic_simulation.py --mock")
        print()

        return 1


if __name__ == "__main__":
    sys.exit(main())
