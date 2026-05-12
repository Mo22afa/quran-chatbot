import json
from pathlib import Path


NOTEBOOKS = [
    Path("mostafa10.ipynb"),
    Path("Quran_Ayah_Correction_Chatbot.ipynb"),
]


results_markdown = """### Expanded Evaluation Results

To make the evaluation stronger, the test set was expanded from a small manual set to **150 generated test cases**.

The expanded test set contains different query types:

| Query Type | Number of Cases |
| :--- | ---: |
| Full ayah without diacritics | 49 |
| First phrase | 27 |
| Middle phrase | 30 |
| Typo phrase | 44 |
| **Total** | **150** |

#### Overall Results on Expanded Test Set

| Metric | Score |
| :--- | ---: |
| **Top-1 Accuracy** | **91.33%** |
| **Recall@5** | **94.67%** |
| **MRR** | **92.63%** |
| **Mean Rank** | **1.07** |
| **nDCG@5** | **93.14%** |

#### Results by Query Type

| Query Type | Cases | Top-1 Accuracy | Recall@5 | MRR | nDCG@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| First phrase | 27 | 100.00% | 100.00% | 100.00% | 100.00% |
| Full ayah | 49 | 97.96% | 97.96% | 97.96% | 97.96% |
| Middle phrase | 30 | 100.00% | 100.00% | 100.00% | 100.00% |
| Typo phrase | 44 | 72.73% | 84.09% | 77.16% | 78.89% |

#### Interpretation

The original `100%` score was based on a very small manual test set. After increasing the test set to 150 cases and adding harder typo-based queries, the result became more realistic.

The system still performs strongly overall, especially for exact, first phrase, and middle phrase inputs. The most difficult case is the `typo_phrase` category, which is expected because these queries contain artificial spelling errors.
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
        if "### Expanded Evaluation Results" not in "".join(cell.get("source", []))
    ]

    insert_at = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.startswith("### External Comparison: Same Base Model Family"):
            insert_at = index
            break

    if insert_at is None:
        insert_at = len(nb["cells"]) - 1

    nb["cells"].insert(insert_at, markdown_cell(results_markdown))
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")
