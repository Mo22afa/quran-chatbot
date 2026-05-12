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

marker = "## 7. Comparison With Previous Work and Baselines"
if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
    print("Comparison section already exists.")
    raise SystemExit(0)

insert_before = None
for index, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if source.strip().startswith("## 7. Audio Feature"):
        insert_before = index
        break

if insert_before is None:
    insert_before = len(nb["cells"])

new_cells = [
    markdown_cell(
        """
## 7. Comparison With Previous Work and Baselines

To compare our results with other work, we must be careful. The same embedding model can produce different scores depending on the dataset, task, language, preprocessing, and evaluation metric.

The model used in this project is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The official Hugging Face model card describes it as a sentence-transformer model that maps sentences and paragraphs to a 384-dimensional vector space and can be used for semantic search and clustering.

However, there is no standard public benchmark specifically for **Quran Ayah Correction** using this exact model. Therefore, direct comparison with unrelated tasks is not scientifically fair.

### Fair Comparison Strategy

We compare three systems on the same Quran test set:

| System | Description |
|---|---|
| Embedding Only | Uses only the Hugging Face semantic embedding model |
| Fuzzy Only | Uses only RapidFuzz text matching |
| Hybrid System | Uses embeddings + fuzzy matching + exact substring priority |

This is a fair comparison because all methods are evaluated on the same dataset and the same test cases.

### External Reference

For external context, the MTEB benchmark evaluates text embedding models across many tasks and languages. MTEB is useful as a general reference for embedding model quality, but it is not a Quran-specific benchmark, so its scores should not be compared directly with our Quran retrieval scores.
"""
    ),
    code_cell(
        """
def find_embedding_only(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    query_embedding = model.encode([clean_input], normalize_embeddings=True)
    semantic_scores = (query_embedding @ ayah_embeddings.T)[0]
    top_indices = np.argsort(semantic_scores)[::-1][:top_k]

    candidates = df.iloc[top_indices].copy()
    candidates["semantic_score"] = semantic_scores[top_indices]
    candidates["final_score"] = candidates["semantic_score"] * 100

    return candidates[
        [
            "surah_no",
            "surah_name",
            "ayah_number",
            "ayah_no_quran",
            "text",
            "semantic_score",
            "final_score",
        ]
    ]


def find_fuzzy_only(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    fuzzy_scores = df["clean_text"].apply(
        lambda text: score_text_match(clean_input, text)
    ).to_numpy()

    candidates = df.copy()
    candidates["fuzzy_score"] = fuzzy_scores
    candidates["match_priority"] = candidates["clean_text"].apply(
        lambda text: match_priority(clean_input, text)
    )
    candidates["final_score"] = candidates["fuzzy_score"]

    candidates = candidates.sort_values(
        ["match_priority", "final_score"], ascending=False
    ).head(top_k)

    return candidates[
        [
            "surah_no",
            "surah_name",
            "ayah_number",
            "ayah_no_quran",
            "text",
            "fuzzy_score",
            "final_score",
        ]
    ]
"""
    ),
    code_cell(
        """
def evaluate_search_method(method_name, search_function, test_cases, top_k=5):
    rows = []

    for case in test_cases:
        predictions = search_function(case["query"], top_k=top_k).reset_index(drop=True)

        expected_mask = (
            (predictions["surah_name"] == case["surah_name"])
            & (predictions["ayah_number"] == case["ayah_number"])
        ).to_numpy()

        if expected_mask.any():
            rank = int(np.where(expected_mask)[0][0]) + 1
            reciprocal_rank = 1 / rank
            recall_at_k = 1
            ndcg_at_k = 1 / np.log2(rank + 1)
        else:
            rank = None
            reciprocal_rank = 0
            recall_at_k = 0
            ndcg_at_k = 0

        top_prediction = predictions.iloc[0]
        rows.append(
            {
                "method": method_name,
                "query": case["query"],
                "expected": f'{case["surah_name"]}:{case["ayah_number"]}',
                "top_prediction": f'{top_prediction["surah_name"]}:{top_prediction["ayah_number"]}',
                "rank": rank,
                "top1_correct": bool(rank == 1),
                f"recall@{top_k}": recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                f"ndcg@{top_k}": ndcg_at_k,
            }
        )

    details = pd.DataFrame(rows)
    summary = {
        "Method": method_name,
        "Top-1 Accuracy": details["top1_correct"].mean(),
        f"Recall@{top_k}": details[f"recall@{top_k}"].mean(),
        "MRR": details["reciprocal_rank"].mean(),
        "Mean Rank": details["rank"].dropna().mean(),
        f"nDCG@{top_k}": details[f"ndcg@{top_k}"].mean(),
    }

    return details, summary
"""
    ),
    code_cell(
        """
baseline_details = []
baseline_summaries = []

methods = [
    ("Embedding Only", find_embedding_only),
    ("Fuzzy Only", find_fuzzy_only),
    ("Hybrid System", find_closest_ayah),
]

for method_name, search_function in methods:
    details, summary = evaluate_search_method(
        method_name,
        search_function,
        test_cases,
        top_k=5,
    )
    baseline_details.append(details)
    baseline_summaries.append(summary)

comparison_details = pd.concat(baseline_details, ignore_index=True)
comparison_summary = pd.DataFrame(baseline_summaries)

comparison_summary
"""
    ),
    code_cell(
        """
comparison_details
"""
    ),
    code_cell(
        """
comparison_plot = comparison_summary.melt(
    id_vars="Method",
    value_vars=["Top-1 Accuracy", "Recall@5", "MRR", "nDCG@5"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(11, 5))
sns.barplot(data=comparison_plot, x="Metric", y="Score", hue="Method")
plt.ylim(0, 1.05)
plt.title("Comparison Between Embedding Only, Fuzzy Only, and Hybrid System")
plt.show()
"""
    ),
    markdown_cell(
        """
### How to Write This Comparison in the Report

The proposed system was compared against two baselines: an embedding-only search method using the same Hugging Face model, and a fuzzy-only method using RapidFuzz. All methods were evaluated on the same Quran ayah correction test set using ranking metrics such as Top-1 Accuracy, Recall@5, MRR, Mean Rank, and nDCG@5.

This comparison is more reliable than comparing directly with generic public benchmark scores because Quran ayah correction is a domain-specific retrieval task.
"""
    ),
]

nb["cells"][insert_before:insert_before] = new_cells
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
