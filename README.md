# Assignment 3 – Information Extraction

This project contains experiments for:
- Medical Named Entity Recognition (NER) on Swedish medical text
- Swedish sentiment classification

The experiments use encoder-only transformer models fine-tuned with the Hugging Face Transformers library.

---

## Requirements

- Python 3.10 or later
- (Optional) NVIDIA GPU with CUDA support for faster training

---

## Environment Setup

The virtual environment is not included in the repository and must be created locally.

### 1. Create a virtual environment

From the project root directory:

**Windows**
```bash
python -m venv venv
```

### 2. Activate the virtual environment

Windows (PowerShell)

```bash
venv\Scripts\activate
```
Linux / macOS
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

notebooks/ – Jupyter notebooks for Task 1 (NER) and Task 2 (Sentiment)

src/ – Helper functions used by the notebooks

results/ – Final evaluation metrics (CSV files)

outputs/ – Training outputs and checkpoints (ignored by Git)

## Notes

Model checkpoints and training outputs are not tracked in Git.

Results can be reproduced by rerunning the notebooks.

GPU acceleration is used automatically if available.
