import json
from pathlib import Path


NOTEBOOK_PATH = Path("Quran_Ayah_Correction_Chatbot.ipynb")


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

new_markdown = """
### External Comparison With People Using the Same Base Model

This section compares our results with public self-reported results from models that use the same base model or fine-tuned versions of it:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The official model maps text into a 384-dimensional dense vector space and is commonly used for semantic search and clustering.

Important note:

> These results are not directly comparable because each system was evaluated on a different dataset and task. They are used as an external reference only.

The fair comparison inside this project is still the comparison between:

```text
Embedding Only
Fuzzy Only
Hybrid System
```

on the same Quran test set.
"""

new_code = """
external_results = pd.DataFrame(
    [
        {
            "System": "ntAnh-dev MiniLM",
            "Task / Dataset": "IR dataset",
            "Accuracy@1": 82.43,
            "Accuracy@5": 97.30,
            "Recall@5": 58.40,
            "MRR@10": 88.01,
            "nDCG@10": 81.75,
            "Source": "https://huggingface.co/ntAnh-dev/paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "System": "ntAnh-dev MiniLM - Eval 2",
            "Task / Dataset": "IR dataset",
            "Accuracy@1": 88.99,
            "Accuracy@5": 95.41,
            "Recall@5": 54.14,
            "MRR@10": 91.64,
            "nDCG@10": 80.43,
            "Source": "https://huggingface.co/ntAnh-dev/paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "System": "SMARTICT ft-tr-rag-v1",
            "Task / Dataset": "Turkish RAG IR",
            "Accuracy@1": 55.97,
            "Accuracy@5": 71.41,
            "Recall@5": 71.41,
            "MRR@10": 62.63,
            "nDCG@10": 65.73,
            "Source": "https://huggingface.co/SMARTICT/paraphrase-multilingual-MiniLM-L12-v2-ft-tr-rag-v1",
        },
        {
            "System": "yahyaabd v1-2",
            "Task / Dataset": "Static table IR",
            "Accuracy@1": 89.90,
            "Accuracy@5": 98.05,
            "Recall@5": 78.96,
            "MRR@10": 93.62,
            "nDCG@10": 82.42,
            "Source": "https://huggingface.co/yahyaabd/paraphrase-multilingual-miniLM-L12-V2-v1-2",
        },
        {
            "System": "Our Hybrid System",
            "Task / Dataset": "Quran Ayah Correction",
            "Accuracy@1": 100.00,
            "Accuracy@5": 100.00,
            "Recall@5": 100.00,
            "MRR@10": 100.00,
            "nDCG@10": 100.00,
            "Source": "Current project evaluation",
        },
    ]
)

external_results
"""

new_plot = """
external_plot = external_results.melt(
    id_vars=["System", "Task / Dataset", "Source"],
    value_vars=["Accuracy@1", "Accuracy@5", "Recall@5", "MRR@10", "nDCG@10"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(13, 6))
sns.barplot(data=external_plot, x="Metric", y="Score", hue="System")
plt.ylim(0, 105)
plt.title("External Comparison With Systems Using the Same Base Model")
plt.ylabel("Score (%)")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.show()
"""

new_interpretation = """
### Interpretation of the External Comparison

Our system achieved `100%` on the current Quran Ayah Correction test set. This is higher than the public self-reported results listed above.

However, this does not mean that our system is universally better than those systems. The reason is that the external systems were evaluated on different datasets and different retrieval tasks.

The correct conclusion is:

> Our hybrid system achieved very strong results on the Quran Ayah Correction task. Public systems based on the same multilingual MiniLM family report Accuracy@1 values between 55.97% and 89.90% on their own datasets. A direct comparison is not fully fair because the datasets and tasks are different.
"""

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if source.startswith("### External Comparison With People Using the Same Base Model"):
        nb["cells"][i]["source"] = new_markdown.strip("\n").splitlines(keepends=True)
    elif source.startswith("external_results = pd.DataFrame("):
        nb["cells"][i]["source"] = new_code.strip("\n").splitlines(keepends=True)
    elif source.startswith("external_plot = external_results.melt("):
        nb["cells"][i]["source"] = new_plot.strip("\n").splitlines(keepends=True)
    elif source.startswith("### Interpretation"):
        nb["cells"][i]["source"] = new_interpretation.strip("\n").splitlines(keepends=True)

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
