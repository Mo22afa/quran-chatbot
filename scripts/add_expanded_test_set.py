import json
from pathlib import Path


NOTEBOOKS = [
    Path("mostafa10.ipynb"),
    Path("Quran_Ayah_Correction_Chatbot.ipynb"),
]


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


expanded_cells = [
    markdown_cell(
        """
### Expanded Test Set

The first manual test set is small. To get a more realistic evaluation, we create a larger automatic test set from the Quran dataset.

The generated test set includes:

- Full ayah without diacritics.
- First phrase from the ayah.
- Middle phrase from long ayahs.
- A noisy phrase with a small character deletion.

Ambiguous exact phrases that appear in multiple ayahs are filtered when possible.
"""
    ),
    code_cell(
        """
def make_typo_query(words):
    words = words.copy()
    if not words:
        return ""

    longest_index = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[longest_index]

    if len(word) <= 4:
        return " ".join(words)

    remove_index = len(word) // 2
    words[longest_index] = word[:remove_index] + word[remove_index + 1:]
    return " ".join(words)


def is_unique_substring(query, corpus_texts):
    return sum(query in text for text in corpus_texts) == 1


def create_expanded_test_cases(dataset, sample_size=80, max_cases=150, random_state=42):
    rng_df = dataset.sample(sample_size, random_state=random_state).reset_index(drop=True)
    corpus_texts = dataset["clean_text"].tolist()
    cases = []
    seen = set()

    for _, row in rng_df.iterrows():
        clean_text = row["clean_text"]
        words = clean_text.split()
        if len(words) < 3:
            continue

        candidate_queries = []

        # Full ayah query without diacritics
        candidate_queries.append(("full_ayah", clean_text))

        # First phrase
        first_phrase = " ".join(words[: min(6, len(words))])
        candidate_queries.append(("first_phrase", first_phrase))

        # Middle phrase for longer ayahs
        if len(words) >= 8:
            start = max(0, len(words) // 2 - 2)
            middle_phrase = " ".join(words[start : start + 5])
            candidate_queries.append(("middle_phrase", middle_phrase))

        # Noisy phrase with a small typo
        noisy_base = words[: min(6, len(words))]
        noisy_phrase = make_typo_query(noisy_base)
        candidate_queries.append(("typo_phrase", noisy_phrase))

        for case_type, query in candidate_queries:
            query = query.strip()
            if len(query.split()) < 3:
                continue

            key = (query, int(row["ayah_no_quran"]))
            if key in seen:
                continue

            if case_type in ["first_phrase", "middle_phrase"] and not is_unique_substring(query, corpus_texts):
                continue

            cases.append(
                {
                    "query": query,
                    "query_type": case_type,
                    "surah_name": row["surah_name"],
                    "ayah_number": int(row["ayah_number"]),
                    "ayah_no_quran": int(row["ayah_no_quran"]),
                }
            )
            seen.add(key)

            if len(cases) >= max_cases:
                return pd.DataFrame(cases)

    return pd.DataFrame(cases)


expanded_test_df = create_expanded_test_cases(
    retrieval_dataset,
    sample_size=80,
    max_cases=150,
    random_state=42,
)

expanded_test_cases = expanded_test_df.to_dict("records")

print("Expanded test cases:", len(expanded_test_cases))
expanded_test_df["query_type"].value_counts()
"""
    ),
    code_cell(
        """
expanded_test_df.head(10)
"""
    ),
]


expanded_eval_cells = [
    markdown_cell(
        """
### Evaluation on Expanded Test Set

Now we evaluate the retrieval system on the larger generated test set. This gives a stronger estimate than the small manual test set.
"""
    ),
    code_cell(
        """
expanded_details, expanded_summary = evaluate_retrieval(expanded_test_cases, top_k=5)
expanded_details.head(10)
"""
    ),
    code_cell(
        """
expanded_summary
"""
    ),
    code_cell(
        """
expanded_type_results = expanded_details.groupby("query").size().reset_index(name="count")

expanded_details_with_type = expanded_details.merge(
    expanded_test_df[["query", "query_type"]],
    on="query",
    how="left",
)

expanded_by_type = expanded_details_with_type.groupby("query_type").agg(
    top1_accuracy=("top1_correct", "mean"),
    recall_at_5=("recall@5", "mean"),
    mrr=("reciprocal_rank", "mean"),
    ndcg_at_5=("ndcg@5", "mean"),
    cases=("query", "count"),
).reset_index()

expanded_by_type
"""
    ),
    code_cell(
        """
plot_expanded = expanded_summary[expanded_summary["Metric"] != "Mean Rank"].copy()

plt.figure(figsize=(9, 4))
sns.barplot(data=plot_expanded, x="Metric", y="Score", color="darkcyan")
plt.ylim(0, 1.05)
plt.title("Expanded Test Set Retrieval Metrics")
plt.xticks(rotation=20)
plt.show()
"""
    ),
]


for notebook_path in NOTEBOOKS:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if "### Expanded Test Set" not in "".join(cell.get("source", []))
        and "### Evaluation on Expanded Test Set" not in "".join(cell.get("source", []))
    ]

    split_insert = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.strip().startswith("## 5️⃣ Modeling"):
            split_insert = index
            break

    if split_insert is None:
        split_insert = len(nb["cells"])

    nb["cells"][split_insert:split_insert] = expanded_cells

    eval_insert = None
    for index, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if source.strip().startswith("### External Comparison: Same Base Model Family"):
            eval_insert = index
            break

    if eval_insert is None:
        eval_insert = len(nb["cells"]) - 1

    nb["cells"][eval_insert:eval_insert] = expanded_eval_cells

    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")
