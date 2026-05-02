# -*- coding: utf-8 -*-

import os
import os.path as osp
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = osp.normpath(osp.dirname(__file__))

LLM_DIR = Path(BASE_DIR).parent.parent / "Dokumente" / "LLaMa_Mistral"
LLM_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("Speicherpfad:", LLM_DIR)

print("\nLade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

print("Lade Modell (das dauert!)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype="auto"
)


# Lokal speichern
print("\nSpeichere lokal...")
tokenizer.save_pretrained(LLM_DIR)
model.save_pretrained(LLM_DIR)

print("\nModell Pfad:")
print(LLM_DIR)