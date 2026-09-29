"""Build Deeb_LLM_Chatbot.ipynb from its percent-format source (standard library only).

Usage:  python tools/build_notebook.py

Cells in the source start with "# %%" (code) or "# %% [markdown]" (markdown, each line prefixed "# ").
Anything before the first marker is ignored. Rebuilding produces a notebook without outputs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook_src" / "Deeb_LLM_Chatbot.py"
OUT = ROOT / "Deeb_LLM_Chatbot.ipynb"


def parse_cells(text):
    cells, current = [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if current:
                cells.append(current)
            current = {"type": "markdown" if "[markdown]" in line else "code", "lines": []}
        elif current is not None:
            if current["type"] == "markdown":
                line = line[2:] if line.startswith("# ") else line.lstrip("#")
            current["lines"].append(line)
    if current:
        cells.append(current)
    for cell in cells:  # trim blank lines around each cell
        while cell["lines"] and not cell["lines"][0].strip():
            cell["lines"].pop(0)
        while cell["lines"] and not cell["lines"][-1].strip():
            cell["lines"].pop()
    return cells


def to_notebook(cells):
    nb_cells = []
    for cell in cells:
        source = [line + "\n" for line in cell["lines"]]
        if source:
            source[-1] = source[-1].rstrip("\n")
        if cell["type"] == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                             "outputs": [], "source": source})
    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",  # makes Colab offer a GPU runtime automatically
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    cells = parse_cells(SRC.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(to_notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.name} ({len(cells)} cells)")
