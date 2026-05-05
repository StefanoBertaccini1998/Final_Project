"""
check_setup.py — Verifica che l'ambiente sia configurato correttamente.
Esegui con: python src/check_setup.py
"""

import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = ""):
    icon = "✅" if ok else "❌"
    msg = f"{icon} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return ok


def main():
    print("\n=== Verifica ambiente Smart Factory Vision Monitor ===\n")
    all_ok = True

    # Python version
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    all_ok &= check("Python", ok, f"{v.major}.{v.minor}.{v.micro}")

    # OpenCV
    try:
        import cv2
        all_ok &= check("OpenCV", True, cv2.__version__)
    except ImportError:
        all_ok &= check("OpenCV", False, "non installato — pip install opencv-python")

    # PyTorch
    try:
        import torch
        all_ok &= check("PyTorch", True, torch.__version__)
        has_cuda = torch.cuda.is_available()
        check("  GPU CUDA", has_cuda, "disponibile" if has_cuda else "non disponibile (useremo CPU)")
    except ImportError:
        all_ok &= check("PyTorch", False, "non installato — pip install torch torchvision")

    # scikit-learn
    try:
        import sklearn
        all_ok &= check("scikit-learn", True, sklearn.__version__)
    except ImportError:
        all_ok &= check("scikit-learn", False, "non installato — pip install scikit-learn")

    # scikit-image
    try:
        import skimage
        all_ok &= check("scikit-image", True, skimage.__version__)
    except ImportError:
        all_ok &= check("scikit-image", False, "non installato — pip install scikit-image")

    # NumPy
    try:
        import numpy as np
        all_ok &= check("NumPy", True, np.__version__)
    except ImportError:
        all_ok &= check("NumPy", False, "non installato")

    # Matplotlib
    try:
        import matplotlib
        all_ok &= check("Matplotlib", True, matplotlib.__version__)
    except ImportError:
        all_ok &= check("Matplotlib", False, "non installato — pip install matplotlib")

    # Dataset
    print()
    dataset_path = Path("data/mvtec_ad")
    if dataset_path.exists():
        categories = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        all_ok &= check("Dataset MVTec", True, f"{len(categories)} categorie trovate: {', '.join(sorted(categories)[:5])}...")
        metal_nut = dataset_path / "metal_nut"
        all_ok &= check("  Categoria metal_nut", metal_nut.exists(),
                        "trovata" if metal_nut.exists() else "non trovata — scarica il dataset")
    else:
        all_ok &= check("Dataset MVTec", False, f"cartella {dataset_path} non trovata — segui SETUP.md Step 3")

    # Output finale
    print()
    if all_ok:
        print("✅ Setup completato — puoi iniziare!\n")
    else:
        print("⚠️  Risolvi i problemi sopra prima di continuare.\n")
        print("Consulta SETUP.md per le istruzioni dettagliate.\n")


if __name__ == "__main__":
    main()
