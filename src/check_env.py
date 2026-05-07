"""Quick sanity check that the environment is set up correctly.

Run this first thing after opening the Codespace:
    python src/check_env.py
"""

import sys


def check_environment() -> bool:
    """Verify all required libraries are installed and importable."""
    print("=" * 50)
    print("Lane Detection Project — Environment Check")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print()

    libraries = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for module_name, display_name in libraries:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK]   {display_name:15s} {version}")
        except ImportError as e:
            print(f"  [FAIL] {display_name:15s} — {e}")
            all_ok = False

    print()
    if all_ok:
        print("All libraries imported successfully. You are ready to go.")
    else:
        print("Some libraries are missing. Run: pip install -r requirements.txt")

    return all_ok


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
