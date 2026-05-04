import pandas as pd
import json
from pathlib import Path

# Pfad zur CSV-Datei
input_csv = "/home/b/PycharmProjects/ba1pmc/make_CLS/test_500.csv"
output_csv = Path("/home/b/PycharmProjects/ba1pmc/make_CLS/test_500_output.csv")  # Beispielhafter Ausgabepfad

# Einlesen der CSV-Datei
df = pd.read_csv(input_csv, encoding="utf-8")

# Überprüfen, ob die Spalten vorhanden sind, sonst initialisieren
required_columns = [
    "row_id", "Final Label:", "RULE:", "BERT:", "CNN1top3", "CNN2top3", "LaMa:",
    "caption", "BERT_score", "CNN1_scores", "CNN2_scores", "LaMa_conf",
    "Final Scores", "Begründung", "uncertain", "pmc_id"
]

for column in required_columns:
    if column not in df.columns:
        df[column] = None  # oder ein Standardwert, falls gewünscht

# Beispielhafte Initialisierung der Spalten, falls nicht vorhanden
# Hier müsstest du die tatsächlichen Daten einfügen, die du in deiner Logik verwendest
# Beispiel:
# records = {
#     "row_id": [1, 2, 3],
#     "Final Label:": ["label1", "label2", "label3"],
#     ...
# }
# df = pd.DataFrame(records)

# Leere captions behandeln
empty_mask = df["caption"].fillna("").astype(str).str.strip().eq("")
df.loc[empty_mask, "pred_label"] = "unknown"
df.loc[empty_mask, "decision_source"] = "empty_text"
df.loc[empty_mask, "decision_reason"] = "empty_text"

# Verzeichnis für die Ausgabedatei erstellen, falls nicht vorhanden
output_csv.parent.mkdir(parents=True, exist_ok=True)

# CSV speichern
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"\nCSV gespeichert unter:\n{output_csv}")

### **Verteilungsanalysen (angepasst an records-Spalten)**
print("\n===== Verteilung Final Label: =====")
print(df["Final Label:"].value_counts(dropna=False))

print("\n===== Verteilung RULE: =====")
print(df["RULE:"].value_counts(dropna=False))

print("\n===== Verteilung BERT: =====")
print(df["BERT:"].value_counts(dropna=False))

print("\n===== Verteilung LaMa: =====")
print(df["LaMa:"].value_counts(dropna=False))

print("\n===== Verteilung uncertain =====")
print(df["uncertain"].value_counts(dropna=False))