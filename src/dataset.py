"""
dataset.py — MVTec AD Dataset Loader
Carica immagini dal dataset MVTec Anomaly Detection.
Supporta classificazione binaria (good vs defective) e multi-classe per tipo di difetto.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Tuple, List, Optional


# Categorie disponibili nel dataset MVTec AD
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


class MVTecDataset:
    """Loader per il dataset MVTec Anomaly Detection.

    Struttura attesa su disco:
        data/mvtec_ad/{category}/train/good/       <- immagini normali (training)
        data/mvtec_ad/{category}/test/good/        <- normali (test)
        data/mvtec_ad/{category}/test/{defect}/    <- difettose (test)

    Esempi di utilizzo:
        # Classificazione binaria (good=0, defective=1)
        dataset = MVTecDataset("data/mvtec_ad", category="metal_nut")
        X_train, y_train = dataset.load_train()
        X_test, y_test = dataset.load_test()

        # Multi-classe (good, broken_large, broken_small, ...)
        dataset = MVTecDataset("data/mvtec_ad", category="bottle", binary=False)
        X_test, y_test, class_names = dataset.load_test(return_classes=True)
    """

    def __init__(
        self,
        root: str,
        category: str = "metal_nut",
        target_size: Tuple[int, int] = (224, 224),
        binary: bool = True
    ):
        """
        Args:
            root: Percorso alla cartella mvtec_ad (es. "data/mvtec_ad").
            category: Categoria del dataset (es. "metal_nut", "bottle").
            target_size: Dimensione (width, height) a cui ridimensionare le immagini.
            binary: Se True, etichette 0=good / 1=defective.
                    Se False, ogni tipo di difetto ha una classe separata.
        """
        self.root = Path(root)
        self.category = category
        self.target_size = target_size
        self.binary = binary
        self.category_path = self.root / category

        if not self.category_path.exists():
            raise FileNotFoundError(
                f"Categoria '{category}' non trovata in {root}.\n"
                f"Categorie disponibili: {MVTEC_CATEGORIES}"
            )

    def _load_images_from_folder(
        self, folder: Path, label: int, label_name: str
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Carica tutte le immagini da una cartella."""
        images, labels, names = [], [], []
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            images.append(img)
            labels.append(label)
            names.append(label_name)

        return images, labels, names

    def load_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carica le immagini di training (solo 'good').

        Returns:
            X: Array (N, H, W, C) con le immagini.
            y: Array (N,) con le etichette (tutti 0 = good).
        """
        train_good = self.category_path / "train" / "good"
        images, labels, _ = self._load_images_from_folder(train_good, label=0, label_name="good")

        print(f"Training set — {self.category}: {len(images)} immagini (tutte 'good')")
        return np.array(images), np.array(labels)

    def load_test(
        self, return_classes: bool = False
    ) -> Tuple:
        """Carica le immagini di test (good + tutti i tipi di difetto).

        Args:
            return_classes: Se True, restituisce anche la lista dei nomi delle classi.

        Returns:
            X: Array (N, H, W, C) con le immagini.
            y: Array (N,) con le etichette.
            class_names (opzionale): Lista dei nomi delle classi.
        """
        test_path = self.category_path / "test"
        all_images, all_labels, all_names = [], [], []

        # Mappa nome_cartella → indice classe
        class_map = {"good": 0}
        defect_idx = 1

        for subfolder in sorted(test_path.iterdir()):
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name

            if self.binary:
                # Modalità binaria: good=0, qualsiasi difetto=1
                label = 0 if folder_name == "good" else 1
                label_name = folder_name
            else:
                # Modalità multi-classe: ogni tipo di difetto è una classe separata
                if folder_name not in class_map:
                    class_map[folder_name] = defect_idx
                    defect_idx += 1
                label = class_map[folder_name]
                label_name = folder_name

            imgs, lbls, nms = self._load_images_from_folder(subfolder, label, label_name)
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_names.extend(nms)

        X = np.array(all_images)
        y = np.array(all_labels)

        # Statistiche
        unique, counts = np.unique(y, return_counts=True)
        mode = "binaria" if self.binary else "multi-classe"
        print(f"\nTest set — {self.category} [{mode}]:")
        if self.binary:
            labels_display = {0: "good", 1: "defective"}
            for u, c in zip(unique, counts):
                print(f"  Classe {u} ({labels_display[u]}): {c} immagini")
        else:
            inv_map = {v: k for k, v in class_map.items()}
            for u, c in zip(unique, counts):
                print(f"  Classe {u} ({inv_map[u]}): {c} immagini")
        print(f"  Totale: {len(X)} immagini\n")

        if return_classes:
            inv_map = {v: k for k, v in class_map.items()}
            class_names = [inv_map[i] for i in range(len(class_map))]
            return X, y, class_names

        return X, y

    def get_class_names(self) -> List[str]:
        """Restituisce i nomi delle classi presenti nel test set."""
        test_path = self.category_path / "test"
        names = ["good"]
        for subfolder in sorted(test_path.iterdir()):
            if subfolder.is_dir() and subfolder.name != "good":
                names.append(subfolder.name)
        return names if not self.binary else ["good", "defective"]


def normalize_dataset(X: np.ndarray) -> np.ndarray:
    """Normalizza un array di immagini in [0, 1].

    Args:
        X: Array (N, H, W, C) di immagini uint8.

    Returns:
        Array float32 normalizzato.
    """
    return X.astype(np.float32) / 255.0


if __name__ == "__main__":
    # Test rapido del loader — esegui con: python src/dataset.py
    import sys

    dataset_path = "data/mvtec_ad"
    category = "metal_nut"

    print(f"Caricamento dataset: {category}")
    dataset = MVTecDataset(dataset_path, category=category, binary=True)

    X_train, y_train = dataset.load_train()
    X_test, y_test = dataset.load_test()

    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test:  {X_test.shape}")
    print(f"Valori unici y_test: {np.unique(y_test, return_counts=True)}")
