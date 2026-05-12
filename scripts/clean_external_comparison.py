import json
from pathlib import Path


NOTEBOOK_PATH = Path("Quran_Ayah_Correction_Chatbot.ipynb")


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

start = None
end = None
for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if source.startswith("### External Comparison With People Using the Same Base Model"):
        start = i
    if start is not None and source.startswith("#### **📊 Quran Ayah Retrieval Evaluation Metrics Guide**"):
        end = i
        break

if start is None or end is None:
    raise SystemExit("Could not find the external comparison section.")

replacement = [
    markdown_cell(
        """
### External Reference Results

This project uses the base model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Some public Hugging Face model cards report results for fine-tuned versions of the same model family. These results are useful as background references only.

They should not be treated as a direct comparison with our Quran system because:

- The datasets are different.
- The languages are different.
- The retrieval tasks are different.
- The test set sizes are different.
- Our system is hybrid, while some external systems are embedding-only.

Therefore, the fair evaluation for this project is the internal baseline comparison:

```text
Embedding Only vs Fuzzy Only vs Hybrid System
```

on the same Quran Ayah Correction test set.
"""
    ),
    code_cell(
        """
external_reference_results = pd.DataFrame(
    [
        {
            "Model / System": "ntAnh-dev MiniLM",
            "Dataset / Task": "Information Retrieval",
            "Accuracy@1": 82.43,
            "Accuracy@5": 97.30,
            "Recall@5": 58.40,
            "MRR@10": 88.01,
            "nDCG@10": 81.75,
        },
        {
            "Model / System": "ntAnh-dev MiniLM - Eval 2",
            "Dataset / Task": "Information Retrieval",
            "Accuracy@1": 88.99,
            "Accuracy@5": 95.41,
            "Recall@5": 54.14,
            "MRR@10": 91.64,
            "nDCG@10": 80.43,
        },
        {
            "Model / System": "SMARTICT ft-tr-rag-v1",
            "Dataset / Task": "Turkish RAG Retrieval",
            "Accuracy@1": 55.97,
            "Accuracy@5": 71.41,
            "Recall@5": 71.41,
            "MRR@10": 62.63,
            "nDCG@10": 65.73,
        },
        {
            "Model / System": "yahyaabd v1-2",
            "Dataset / Task": "Static Table Retrieval",
            "Accuracy@1": 89.90,
            "Accuracy@5": 98.05,
            "Recall@5": 78.96,
            "MRR@10": 93.62,
            "nDCG@10": 82.42,
        },
    ]
)

external_reference_results
"""
    ),
    markdown_cell(
        """
### Discussion

The external results show that systems based on the same multilingual MiniLM family commonly report `Accuracy@1` values between about `55.97%` and `89.90%` on their own retrieval datasets.

Our system achieved `100%` on the current Quran Ayah Correction test set, but this score should be interpreted carefully because the test set is small and domain-specific.

The correct conclusion is:

> The proposed hybrid system performs strongly on the current Quran test set. However, external results cannot be compared directly because they use different datasets and tasks.
"""
    ),
]

nb["cells"][start:end] = replacement
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Cleaned external comparison in {NOTEBOOK_PATH}")
