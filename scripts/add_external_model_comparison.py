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

marker = "### External Comparison With People Using the Same Base Model"
if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
    print("External comparison section already exists.")
    raise SystemExit(0)

insert_before = None
for index, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "#### **📊 Quran Ayah Retrieval Evaluation Metrics Guide**" in source:
        insert_before = index
        break

if insert_before is None:
    insert_before = len(nb["cells"])

new_cells = [
    markdown_cell(
        """
### External Comparison With People Using the Same Base Model

The model used in this project is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The official Hugging Face model card describes this model as a SentenceTransformer that maps text into a 384-dimensional vector space and can be used for semantic search and clustering.

To compare our results with previous work, we collected public self-reported results from Hugging Face model cards that use the same base model or fine-tuned versions of it.

Important note: these results are not perfectly comparable because each model was evaluated on a different dataset and task. Therefore, this table is used as an external reference, while the fair comparison remains the internal comparison between `Embedding Only`, `Fuzzy Only`, and `Hybrid System` on the same Quran test set.

Sources:

- Official base model: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- ntAnh-dev fine-tuned model: https://huggingface.co/ntAnh-dev/paraphrase-multilingual-MiniLM-L12-v2
- SMARTICT fine-tuned model: https://huggingface.co/SMARTICT/paraphrase-multilingual-MiniLM-L12-v2-ft-tr-rag-v1
- yahyaabd fine-tuned model: https://huggingface.co/yahyaabd/paraphrase-multilingual-miniLM-L12-V2-v1-2
"""
    ),
    code_cell(
        """
external_results = pd.DataFrame(
    [
        {
            "System": "Our Hybrid System",
            "Dataset / Task": "Quran Ayah Correction",
            "Accuracy@1": 1.000,
            "Accuracy@5": 1.000,
            "Recall@5": 1.000,
            "MRR@10": 1.000,
            "nDCG@10": 1.000,
            "Notes": "Our Quran test set; hybrid retrieval system",
        },
        {
            "System": "ntAnh-dev fine-tuned MiniLM",
            "Dataset / Task": "paraphrase multilingual MiniLM L12 v2 251218",
            "Accuracy@1": 0.824,
            "Accuracy@5": 0.973,
            "Recall@5": 0.584,
            "MRR@10": 0.880,
            "nDCG@10": 0.818,
            "Notes": "Self-reported Hugging Face evaluation",
        },
        {
            "System": "SMARTICT ft-tr-rag-v1",
            "Dataset / Task": "dim_384 information retrieval",
            "Accuracy@1": 0.560,
            "Accuracy@5": 0.714,
            "Recall@5": 0.714,
            "MRR@10": 0.626,
            "nDCG@10": 0.657,
            "Notes": "Self-reported Hugging Face evaluation",
        },
        {
            "System": "yahyaabd v1-2",
            "Dataset / Task": "bps statictable ir",
            "Accuracy@1": 0.899,
            "Accuracy@5": 0.980,
            "Recall@5": 0.790,
            "MRR@10": 0.936,
            "nDCG@10": 0.824,
            "Notes": "Self-reported Hugging Face evaluation",
        },
    ]
)

external_results
"""
    ),
    code_cell(
        """
external_plot = external_results.melt(
    id_vars=["System", "Dataset / Task", "Notes"],
    value_vars=["Accuracy@1", "Accuracy@5", "Recall@5", "MRR@10", "nDCG@10"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(13, 6))
sns.barplot(data=external_plot, x="Metric", y="Score", hue="System")
plt.ylim(0, 1.05)
plt.title("External Comparison With Systems Using the Same Base Model")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.show()
"""
    ),
    markdown_cell(
        """
### Interpretation

Our system achieves strong scores on the Quran ayah correction test set. However, these numbers should not be interpreted as a direct superiority over the external systems because the external models were evaluated on different datasets and tasks.

The correct conclusion is:

> The proposed hybrid method performs strongly on the Quran Ayah Correction task, and its retrieval scores are competitive with public self-reported results from systems based on the same multilingual MiniLM sentence-transformer family. A direct ranking is not fully fair because the datasets and evaluation protocols are different.
"""
    ),
]

nb["cells"][insert_before:insert_before] = new_cells
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
