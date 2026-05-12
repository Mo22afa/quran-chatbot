import json
from pathlib import Path


NOTEBOOKS = [
    Path("mostafa10.ipynb"),
    Path("Quran_Ayah_Correction_Chatbot.ipynb"),
]


summary_markdown = """### Model Results and External Comparison Summary

This table shows the actual performance of our system after expanding the test set to **150 test cases**.

#### Our Model Results

| Metric | Our Result | Evaluation Level |
| :--- | ---: | :--- |
| **Top-1 Accuracy** | **91.33%** | Excellent |
| **Accuracy@5 / Recall@5** | **94.67%** | Excellent |
| **MRR** | **92.63%** | Excellent |
| **Mean Rank** | **1.07** | Excellent |
| **nDCG@5** | **93.14%** | Excellent |

#### Other People Results Using the Same Model Family

| System | Dataset / Task | Accuracy@1 | Accuracy@5 | Recall@5 | MRR@10 | nDCG@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ntAnh-dev MiniLM | Information Retrieval | 82.43% | 97.30% | 58.40% | 88.01% | 81.75% |
| ntAnh-dev MiniLM - Eval 2 | Information Retrieval | 88.99% | 95.41% | 54.14% | 91.64% | 80.43% |
| SMARTICT ft-tr-rag-v1 | Turkish RAG Retrieval | 55.97% | 71.41% | 71.41% | 62.63% | 65.73% |
| yahyaabd v1-2 | Static Table Retrieval | 89.90% | 98.05% | 78.96% | 93.62% | 82.42% |

#### Are Our Results Good?

| Metric | Weak Result | Good (Lab Project) | Excellent | Our Result |
| :--- | :--- | :--- | :--- | ---: |
| **Top-1 Accuracy** | < 60% | 75% - 90% | 90%+ | **91.33%** |
| **Accuracy@5 / Recall@5** | < 60% | 75% - 90% | 90%+ | **94.67%** |
| **MRR** | < 50% | 70% - 85% | 85%+ | **92.63%** |
| **nDCG@5** | < 60% | 75% - 90% | 90%+ | **93.14%** |

#### Important Note

The external results are not directly comparable because they were evaluated on different datasets and tasks. They are used only as a reference for the same MiniLM sentence-transformer model family.

The most reliable result for this project is our expanded evaluation on **150 Quran Ayah Correction test cases**.
"""


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


for notebook_path in NOTEBOOKS:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if "### Model Results and External Comparison Summary" not in "".join(cell.get("source", []))
    ]

    insert_at = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.startswith("#### 📝 Summary Table: Are your results good?"):
            insert_at = index
            break

    if insert_at is None:
        for index, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", []))
            if source.startswith("## 6️⃣ Testing"):
                insert_at = index
                break

    if insert_at is None:
        insert_at = len(nb["cells"]) - 1

    nb["cells"].insert(insert_at, markdown_cell(summary_markdown))
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")
