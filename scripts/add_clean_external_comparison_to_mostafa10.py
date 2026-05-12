import json
from pathlib import Path


FILES = [
    Path("mostafa10.ipynb"),
    Path("Quran_Ayah_Correction_Chatbot.ipynb"),
]


comparison_markdown = """### External Comparison: Same Base Model Family

The base model used in this project is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

To evaluate our system more clearly, we compare our results with public reported results from systems that used the same model family or fine-tuned versions of it.

> Important: this is not a perfect direct comparison because the datasets and tasks are different. It is used only as an external reference.

| System | Dataset / Task | Accuracy@1 | Accuracy@5 | Recall@5 | MRR@10 | nDCG@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ntAnh-dev MiniLM | Information Retrieval | 82.43% | 97.30% | 58.40% | 88.01% | 81.75% |
| ntAnh-dev MiniLM - Eval 2 | Information Retrieval | 88.99% | 95.41% | 54.14% | 91.64% | 80.43% |
| SMARTICT ft-tr-rag-v1 | Turkish RAG Retrieval | 55.97% | 71.41% | 71.41% | 62.63% | 65.73% |
| yahyaabd v1-2 | Static Table Retrieval | 89.90% | 98.05% | 78.96% | 93.62% | 82.42% |
| **Our Hybrid System** | **Quran Ayah Correction** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

#### Interpretation

Our hybrid system achieved `100%` on the current Quran Ayah Correction test set. This result is strong, but it should be interpreted carefully because our test set is small and domain-specific.

The external systems report `Accuracy@1` between `55.97%` and `89.90%` on their own datasets. Since the datasets and tasks are different, this table should be used as a reference, not as a strict ranking.

The fair internal comparison in this project is:

```text
Embedding Only vs Fuzzy Only vs Hybrid System
```

because all three methods are evaluated on the same Quran dataset and the same test cases.

Sources:

- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/ntAnh-dev/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/SMARTICT/paraphrase-multilingual-MiniLM-L12-v2-ft-tr-rag-v1
- https://huggingface.co/yahyaabd/paraphrase-multilingual-miniLM-L12-V2-v1-2
"""


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


for notebook_path in FILES:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if "External Comparison: Same Base Model Family" not in "".join(cell.get("source", []))
    ]

    insert_at = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.startswith("#### 📝 Summary Table: Are your results good?"):
            insert_at = index
            break

    if insert_at is None:
        insert_at = len(nb["cells"]) - 1

    nb["cells"].insert(insert_at, markdown_cell(comparison_markdown))
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")
