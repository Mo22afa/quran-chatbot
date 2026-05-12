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

marker = "### Additional Retrieval Metrics"
if any(marker in "".join(cell.get("source", [])) for cell in nb["cells"]):
    print("Additional retrieval metrics section already exists.")
    raise SystemExit(0)

insert_after = None
for index, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "#### Evaluation Metrics Guide" in source:
        insert_after = index + 1
        break

if insert_after is None:
    insert_after = len(nb["cells"])

new_cells = [
    markdown_cell(
        """
### Additional Retrieval Metrics Beyond Accuracy

Accuracy is useful, but this project is a retrieval system, not a normal classification model. Therefore, we can evaluate the model using ranking-based metrics:

| Metric | Meaning |
|---|---|
| `Precision@K` | How many of the top K returned results are relevant |
| `Recall@K` | Whether the correct ayah appears in the top K results |
| `MRR` | Rewards the system when the correct ayah appears at a higher rank |
| `Mean Rank` | Average position of the correct ayah in the returned results |
| `nDCG@K` | Ranking quality metric that gives higher value when the correct result appears near the top |

For this project, `MRR`, `Mean Rank`, and `nDCG@K` are more informative than simple accuracy.
"""
    ),
    code_cell(
        """
def evaluate_retrieval(test_cases, top_k=5):
    rows = []

    for case in test_cases:
        predictions = find_closest_ayah(case["query"], top_k=top_k).reset_index(drop=True)

        expected_mask = (
            (predictions["surah_name"] == case["surah_name"])
            & (predictions["ayah_number"] == case["ayah_number"])
        ).to_numpy()

        if expected_mask.any():
            rank = int(np.where(expected_mask)[0][0]) + 1
            reciprocal_rank = 1 / rank
            precision_at_k = 1 / top_k
            recall_at_k = 1
            ndcg_at_k = 1 / np.log2(rank + 1)
        else:
            rank = None
            reciprocal_rank = 0
            precision_at_k = 0
            recall_at_k = 0
            ndcg_at_k = 0

        top_prediction = predictions.iloc[0]
        rows.append(
            {
                "query": case["query"],
                "expected": f'{case["surah_name"]}:{case["ayah_number"]}',
                "top_prediction": f'{top_prediction["surah_name"]}:{top_prediction["ayah_number"]}',
                "rank": rank,
                f"precision@{top_k}": precision_at_k,
                f"recall@{top_k}": recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                f"ndcg@{top_k}": ndcg_at_k,
                "top_score": round(float(top_prediction["final_score"]), 2),
            }
        )

    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "Metric": [
                f"Mean Precision@{top_k}",
                f"Mean Recall@{top_k}",
                "MRR",
                "Mean Rank",
                f"Mean nDCG@{top_k}",
            ],
            "Score": [
                details[f"precision@{top_k}"].mean(),
                details[f"recall@{top_k}"].mean(),
                details["reciprocal_rank"].mean(),
                details["rank"].dropna().mean(),
                details[f"ndcg@{top_k}"].mean(),
            ],
        }
    )

    return details, summary
"""
    ),
    code_cell(
        """
retrieval_details, retrieval_summary = evaluate_retrieval(test_cases, top_k=5)
retrieval_details
"""
    ),
    code_cell(
        """
retrieval_summary
"""
    ),
    code_cell(
        """
plt.figure(figsize=(9, 4))
plot_df = retrieval_summary[retrieval_summary["Metric"] != "Mean Rank"].copy()
sns.barplot(data=plot_df, x="Metric", y="Score", color="mediumpurple")
plt.ylim(0, 1.05)
plt.title("Retrieval Metrics Beyond Accuracy")
plt.xticks(rotation=20)
plt.show()
"""
    ),
]

nb["cells"][insert_after:insert_after] = new_cells
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
