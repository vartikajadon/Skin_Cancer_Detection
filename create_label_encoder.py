# create_label_encoder.py
"""Utility to create a simple label encoder pickle for the skin lesion model.
The mapping follows the canonical class indices used throughout the project.
"""
import json
import pathlib

label_mapping = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

# Save mapping as JSON for the predictor
output_path = pathlib.Path(__file__).resolve().parent / "models" / "label_encoder.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4)
print(f"Label encoder JSON saved to {output_path}")
