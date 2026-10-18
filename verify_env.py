"""Environment and dependency verification script.

Validates that all deep learning, federated learning, NLP, computer vision,
and API dependencies are properly installed and ready for execution.
"""

import sys
import platform
from pathlib import Path


def check_module(name, import_fn):
    """Attempt to import a module and return (status_bool, version_or_error)."""
    try:
        mod = import_fn()
        version = getattr(mod, "__version__", "Available")
        return True, str(version)
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print(" PRIVACY-PRESERVING MULTIMODAL NEWS RECOMMENDATION")
    print(" Environment & Dependency Verification (Phase 1)")
    print("=" * 70)

    # 1. System Details
    print(f"\n[System Information]")
    print(f"  OS Platform       : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python Version    : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  Project Root      : {Path(__file__).parent.resolve()}")

    # 2. Dependency Checks
    checks = [
        ("PyTorch (Core DL)", lambda: __import__("torch")),
        ("Torchvision (Vision)", lambda: __import__("torchvision")),
        ("Transformers (DistilBERT)", lambda: __import__("transformers")),
        ("Flower (FL Orchestrator)", lambda: __import__("flwr")),
        ("FastAPI (Backend API)", lambda: __import__("fastapi")),
        ("Uvicorn (ASGI Server)", lambda: __import__("uvicorn")),
        ("Pydantic (Validation)", lambda: __import__("pydantic")),
        ("NumPy (Numerics)", lambda: __import__("numpy")),
        ("Pandas (DataFrames)", lambda: __import__("pandas")),
        ("Scikit-Learn (Metrics)", lambda: __import__("sklearn")),
        ("SciPy (Stats & Science)", lambda: __import__("scipy")),
        ("PyYAML (Config Loader)", lambda: __import__("yaml")),
        ("Pillow (Image Processing)", lambda: __import__("PIL")),
        ("Requests (Networking)", lambda: __import__("requests")),
        ("Matplotlib (Visuals)", lambda: __import__("matplotlib")),
        ("Pytest (Unit Testing)", lambda: __import__("pytest")),
    ]

    print(f"\n[Dependency Status]")
    all_passed = True
    missing_packages = []

    for name, import_fn in checks:
        ok, detail = check_module(name, import_fn)
        status_tag = "[PASS]" if ok else "[FAIL]"
        print(f"  {status_tag:<8} {name:<28} : {detail}")
        if not ok:
            all_passed = False
            missing_packages.append(name.split()[0].lower())

    # 3. Hardware Acceleration & Device Verification
    print(f"\n[Hardware Acceleration & Tensor Compute]")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
        device = torch.device("cuda" if cuda_available else "cpu")
        print(f"  CUDA Available    : {cuda_available}")
        print(f"  Selected Device   : {device} ({device_name})")

        # Tensor verification test
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.matmul(a, b)
        print(f"  Tensor Matmul Test: SUCCESS (Shape: {list(c.shape)}, Device: {c.device})")
    except Exception as e:
        print(f"  Tensor Matmul Test: FAILED ({e})")
        all_passed = False

    # 4. Configuration Integrity Check
    print(f"\n[Configuration Integrity Check]")
    try:
        import yaml

        root = Path(__file__).parent
        configs = [
            root / "configs" / "base_config.yaml",
            root / "configs" / "federated_config.yaml",
            root / "configs" / "privacy_config.yaml",
        ]
        for cfg_path in configs:
            if not cfg_path.exists():
                raise FileNotFoundError(f"Missing config file: {cfg_path.name}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            print(f"  [PASS] {cfg_path.name:<25} : Successfully parsed ({len(data)} top-level sections)")
    except Exception as e:
        print(f"  [FAIL] Configuration loading error: {e}")
        all_passed = False

    # 5. Summary Conclusion
    print("\n" + "=" * 70)
    if all_passed:
        print("  RESULT: PHASE 1 ENVIRONMENT SETUP COMPLETE & VERIFIED")
        print("  All essential frameworks and tools are available.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("  RESULT: INCOMPLETE ENVIRONMENT")
        print(f"  Some dependencies are missing. Run:")
        print(f"    pip install -r requirements.txt")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
