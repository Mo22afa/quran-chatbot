import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matcher import QuranMatcher
from preprocessing import normalize_arabic


def make_typo_query(words):
    words = words.copy()
    if not words:
        return ""

    longest_index = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[longest_index]

    if len(word) <= 4:
        return " ".join(words)

    remove_index = len(word) // 2
    words[longest_index] = word[:remove_index] + word[remove_index + 1 :]
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

        candidate_queries = [("full_ayah", clean_text)]

        first_phrase = " ".join(words[: min(6, len(words))])
        candidate_queries.append(("first_phrase", first_phrase))

        if len(words) >= 8:
            start = max(0, len(words) // 2 - 2)
            middle_phrase = " ".join(words[start : start + 5])
            candidate_queries.append(("middle_phrase", middle_phrase))

        noisy_base = words[: min(6, len(words))]
        noisy_phrase = make_typo_query(noisy_base)
        candidate_queries.append(("typo_phrase", noisy_phrase))

        for case_type, query in candidate_queries:
            query = normalize_arabic(query).strip()
            if len(query.split()) < 3:
                continue

            key = (query, int(row["ayah_no_quran"]))
            if key in seen:
                continue

            if case_type in ["first_phrase", "middle_phrase"] and not is_unique_substring(
                query, corpus_texts
            ):
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


def evaluate(matcher, test_cases, top_k=5):
    rows = []

    for index, case in enumerate(test_cases, 1):
        predictions = matcher.search(case["query"], top_k=top_k)

        matches = [
            rank
            for rank, pred in enumerate(predictions, 1)
            if pred["surah_name"] == case["surah_name"]
            and int(pred["ayah_number"]) == int(case["ayah_number"])
        ]

        rank = matches[0] if matches else None
        top_prediction = predictions[0]

        rows.append(
            {
                "query": case["query"],
                "query_type": case["query_type"],
                "expected": f'{case["surah_name"]}:{case["ayah_number"]}',
                "top_prediction": f'{top_prediction["surah_name"]}:{top_prediction["ayah_number"]}',
                "rank": rank,
                "top1_correct": rank == 1,
                f"recall@{top_k}": int(rank is not None),
                "reciprocal_rank": 1 / rank if rank else 0,
                f"ndcg@{top_k}": 1 / np.log2(rank + 1) if rank else 0,
                "top_score": round(float(top_prediction["final_score"]), 2),
            }
        )

        if index % 25 == 0:
            print(f"Evaluated {index}/{len(test_cases)}")

    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "Metric": [
                "Test Cases",
                "Top-1 Accuracy",
                f"Recall@{top_k}",
                "MRR",
                "Mean Rank",
                f"nDCG@{top_k}",
            ],
            "Score": [
                len(details),
                details["top1_correct"].mean(),
                details[f"recall@{top_k}"].mean(),
                details["reciprocal_rank"].mean(),
                details["rank"].dropna().mean(),
                details[f"ndcg@{top_k}"].mean(),
            ],
        }
    )
    by_type = (
        details.groupby("query_type")
        .agg(
            cases=("query", "count"),
            top1_accuracy=("top1_correct", "mean"),
            recall_at_5=(f"recall@{top_k}", "mean"),
            mrr=("reciprocal_rank", "mean"),
            ndcg_at_5=(f"ndcg@{top_k}", "mean"),
        )
        .reset_index()
    )
    return details, summary, by_type


def main():
    matcher = QuranMatcher("data/quran.csv")
    expanded_test_df = create_expanded_test_cases(
        matcher.df,
        sample_size=80,
        max_cases=150,
        random_state=42,
    )
    print(f"Expanded test cases: {len(expanded_test_df)}")
    print(expanded_test_df["query_type"].value_counts().to_string())

    details, summary, by_type = evaluate(matcher, expanded_test_df.to_dict("records"), top_k=5)

    out_dir = "evaluation"
    os.makedirs(out_dir, exist_ok=True)
    expanded_test_df.to_csv(f"{out_dir}/expanded_test_cases.csv", index=False, encoding="utf-8")
    details.to_csv(f"{out_dir}/expanded_evaluation_details.csv", index=False, encoding="utf-8")
    summary.to_csv(f"{out_dir}/expanded_evaluation_summary.csv", index=False, encoding="utf-8")
    by_type.to_csv(f"{out_dir}/expanded_evaluation_by_type.csv", index=False, encoding="utf-8")

    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nBy query type")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    main()
