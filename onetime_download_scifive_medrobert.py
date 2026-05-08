from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5Tokenizer,
    T5ForConditionalGeneration
)

# =========================================================
# ZIELORDNER
# =========================================================

BASE_DIR = Path.home() / "Dokumente"

MEDROBERTA_DIR = BASE_DIR / "medroberta"

SCIFIVE_DIR = BASE_DIR / "SciFive_Pubmed_PMC"

print("MEDROBERTA_DIR:", MEDROBERTA_DIR)
print("SCIFIVE_DIR:", SCIFIVE_DIR)

# =========================================================
# MODEL NAMEN
# =========================================================

MEDROBERTA_NAME = "FremyCompany/BioLORD-2023-M"

SCIFIVE_NAME = "razent/SciFive-large-Pubmed_PMC"

# =========================================================
# ORDNER ERSTELLEN
# =========================================================

MEDROBERTA_DIR.mkdir(parents=True, exist_ok=True)

SCIFIVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# MEDROBERTA DOWNLOAD
# =========================================================

print("\n===================================================")
print("DOWNLOAD MEDROBERTA")
print("===================================================\n")

med_tokenizer = AutoTokenizer.from_pretrained(
    MEDROBERTA_NAME
)

med_model = AutoModel.from_pretrained(
    MEDROBERTA_NAME
)

print("Speichere MedRoBERTa lokal...")

med_tokenizer.save_pretrained(MEDROBERTA_DIR)

med_model.save_pretrained(MEDROBERTA_DIR)

print("MedRoBERTa gespeichert unter:")
print(MEDROBERTA_DIR)

# =========================================================
# SCIFIVE DOWNLOAD
# =========================================================

print("\n===================================================")
print("DOWNLOAD SCIFIVE")
print("===================================================\n")

scifive_tokenizer = T5Tokenizer.from_pretrained(
    SCIFIVE_NAME
)

scifive_model = T5ForConditionalGeneration.from_pretrained(
    SCIFIVE_NAME
)

print("Speichere SciFive lokal...")

scifive_tokenizer.save_pretrained(SCIFIVE_DIR)

scifive_model.save_pretrained(SCIFIVE_DIR)

print("SciFive gespeichert unter:")
print(SCIFIVE_DIR)

# =========================================================
# FERTIG
# =========================================================

print("\n===================================================")
print("ALLE MODELLE ERFOLGREICH GESPEICHERT")
print("===================================================\n")