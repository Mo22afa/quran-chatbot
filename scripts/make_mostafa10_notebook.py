import json
from pathlib import Path


SOURCE = Path("Quran_Ayah_Correction_Chatbot.ipynb")
TARGET = Path("mostafa10.ipynb")


summary_table = """#### 📝 Summary Table: Are your results good?

| Metric | Weak Result | Good (Lab Project) | Excellent (SOTA) |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | < 0.60 | 0.75 - 0.90 | 0.90+ |
| **Recall@5** | < 0.60 | 0.75 - 0.90 | 0.90+ |
| **MRR** | < 0.50 | 0.70 - 0.85 | 0.85+ |
| **nDCG@5** | < 0.60 | 0.75 - 0.90 | 0.90+ |

For this Quran ayah retrieval project, **MRR** and **nDCG@5** are more useful than simple accuracy because they evaluate the ranking quality of the returned ayahs.
"""


nb = json.loads(SOURCE.read_text(encoding="utf-8"))

for index, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if source.startswith("table = \"\"\"") and "Top-1 Accuracy" in source:
        nb["cells"][index] = {
            "cell_type": "markdown",
            "metadata": {},
            "source": summary_table.strip("\n").splitlines(keepends=True),
        }
        break
else:
    insert_at = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.startswith("#### **📊 Quran Ayah Retrieval Evaluation Metrics Guide**"):
            insert_at = index
            break
    if insert_at is None:
        insert_at = len(nb["cells"])
    nb["cells"].insert(
        insert_at,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": summary_table.strip("\n").splitlines(keepends=True),
        },
    )

TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
SOURCE.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Updated {SOURCE}")
print(f"Created {TARGET}")
