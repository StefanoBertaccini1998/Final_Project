# Setup Guide — Smart Factory Vision Monitor

Segui questi passi nell'ordine esatto. Tempo stimato: 30-45 minuti.

---

## Step 1 — Crea l'ambiente virtuale

Apri il terminale in VS Code (`Ctrl + `` `) nella cartella del progetto:

```bash
# Crea ambiente virtuale
python -m venv venv

# Attivalo (Windows)
venv\Scripts\activate

# Attivalo (Mac/Linux)
source venv/bin/activate
```

Dovresti vedere `(venv)` all'inizio del terminale. Se non ce l'hai, fai prima:
```bash
pip install virtualenv
```

---

## Step 2 — Installa le dipendenze

```bash
pip install -r requirements.txt
```

Ci vogliono 5-10 minuti la prima volta. Se hai errori su `torch`, usa questo comando alternativo:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy scikit-learn scikit-image matplotlib Pillow joblib tqdm
```

> **Nota**: installiamo la versione CPU di PyTorch. Per GPU, vedi la sezione opzionale in fondo.

---

## Step 3 — Scarica il dataset MVTec AD da Kaggle

### 3a — Installa Kaggle CLI
```bash
pip install kaggle
```

### 3b — Configura le credenziali
1. Vai su [kaggle.com](https://www.kaggle.com) → Account → Settings → API → **Create New Token**
2. Scarica il file `kaggle.json`
3. Spostalo nella posizione giusta:
   - **Windows**: `C:\Users\TUO_NOME\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

### 3c — Scarica il dataset
```bash
# Dalla cartella del progetto
kaggle datasets download -d ipythonx/mvtec-ad -p data/ --unzip
```

Il download è ~4GB e ci vuole qualche minuto. La struttura finale sarà:

```
data/
└── mvtec_ad/
    ├── bottle/
    │   ├── train/good/        ← immagini normali per training
    │   └── test/
    │       ├── good/          ← normali per test
    │       ├── broken_large/  ← difettose
    │       └── broken_small/  ← difettose
    ├── metal_nut/             ← useremo questa categoria!
    ├── screw/
    └── ...                    ← altre 12 categorie
```

> **Alternativa senza CLI**: vai su [kaggle.com/datasets/ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad), clicca Download e decomprimi manualmente in `data/`.

---

## Step 4 — Verifica che tutto funzioni

```bash
python src/check_setup.py
```

Se vedi questo output, sei pronto:
```
✅ OpenCV: 4.x.x
✅ PyTorch: 2.x.x
✅ scikit-learn: 1.x.x
✅ Dataset trovato: data/mvtec_ad/metal_nut
✅ Setup completato — puoi iniziare!
```

---

## Struttura del progetto

```
Final_Project/
├── SETUP.md               ← questa guida
├── requirements.txt
├── src/
│   ├── check_setup.py     ← verifica ambiente
│   ├── dataset.py         ← caricamento dati MVTec
│   ├── preprocessing.py   ← preprocessing immagini
│   ├── features.py        ← HOG, SIFT
│   ├── models/
│   │   ├── classical.py   ← SVM + HOG (baseline)
│   │   └── deep.py        ← EfficientNet fine-tuned
│   ├── postprocessing.py  ← zone check + NMS
│   ├── evaluate.py        ← metriche
│   └── main.py            ← script principale
├── notebooks/
│   └── experiments.ipynb  ← esplora i dati qui
├── data/
│   └── mvtec_ad/          ← dataset (non committare su GitHub!)
├── outputs/
│   ├── checkpoints/       ← modelli salvati
│   └── results/           ← plot e metriche
└── docs/
    └── technical_analysis.pdf
```

---

## Opzionale — GPU con CUDA (se hai una NVIDIA)

```bash
# Prima controlla la versione CUDA
nvidia-smi

# Poi installa PyTorch con CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Opzionale — Estensioni VS Code consigliate

- **Python** (Microsoft) — indispensabile
- **Pylance** — autocompletamento avanzato
- **Jupyter** — per i notebook dentro VS Code
- **GitLens** — per gestire Git facilmente
