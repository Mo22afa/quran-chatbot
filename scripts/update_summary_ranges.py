import json
from pathlib import Path


NOTEBOOKS = [
    Path("mostafa10.ipynb"),
    Path("Quran_Ayah_Correction_Chatbot.ipynb"),
]


summary_table = """#### 📝 Summary Table: Are your results good?

| Metric | Weak Result | Average Result | Good Result | Excellent Result |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 0% - 59% | 60% - 74% | 75% - 89% | 90% - 100% |
| **Recall@5** | 0% - 59% | 60% - 74% | 75% - 89% | 90% - 100% |
| **MRR** | 0% - 49% | 50% - 69% | 70% - 84% | 85% - 100% |
| **nDCG@5** | 0% - 59% | 60% - 74% | 75% - 89% | 90% - 100% |

For this Quran ayah retrieval project, **MRR** and **nDCG@5** are more useful than simple accuracy because they evaluate the ranking quality of the returned ayahs.
"""


for notebook_path in NOTEBOOKS:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    replaced = False
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("#### 📝 Summary Table: Are your results good?"):
            cell["cell_type"] = "markdown"
            cell["metadata"] = {}
            cell["source"] = summary_table.strip("\n").splitlines(keepends=True)
            replaced = True

    if not replaced:
        raise SystemExit(f"Summary table not found in {notebook_path}")

    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")
