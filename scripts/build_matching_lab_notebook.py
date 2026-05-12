import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Quran_Ayah_Correction_Chatbot.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md("# Project - Quran Ayah Correction Chatbot **(LAB)** 👨🏻‍💻📖"),
    md("## 1️⃣ Required Libraries"),
    code(
        """
# Basic Libraries
import os
import re
import gc
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NLP / Retrieval Libraries
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# Audio Display
from IPython.display import Audio, display

pd.set_option("display.max_colwidth", 160)
plt.style.use("seaborn-v0_8")
"""
    ),
    md("## 2️⃣ Read Dataset"),
    md("### Read From GitHub"),
    code(
        """
DATA_URL = "https://raw.githubusercontent.com/malekverse/quran-dataset/main/quran_dataset.csv"
"""
    ),
    code(
        """
df_load = pd.read_csv(DATA_URL)
"""
    ),
    md(
        """
#### 📂 **Project Data Source: Quran Dataset**

##### 1. Overview

This project uses a complete Quran dataset. The dataset contains the Arabic Quran text and metadata for each ayah.

##### 2. Important Columns

| Column | Description |
| :--- | :--- |
| `surah_no` | Surah number |
| `surah_name_ar` | Arabic surah name |
| `ayah_no_surah` | Ayah number inside the surah |
| `ayah_no_quran` | Global ayah number in the Quran |
| `ayah_ar` | Arabic ayah text |
| `ayah_en` | English translation |

##### 3. Project Goal

The goal is to build a chatbot that corrects Quran ayah input by retrieving the closest correct ayah from the dataset.

The model is used for retrieval only. It does not generate Quranic text.
"""
    ),
    md("### Split to Columns"),
    code(
        """
df = df_load.rename(
    columns={
        "surah_name_ar": "surah_name",
        "ayah_no_surah": "ayah_number",
        "ayah_ar": "text",
    }
)

df = df[
    [
        "surah_no",
        "surah_name",
        "ayah_number",
        "ayah_no_quran",
        "text",
        "ayah_en",
    ]
].copy()
"""
    ),
    code(
        """
# Keep a local copy for the Streamlit chatbot app
os.makedirs("data", exist_ok=True)
df.to_csv("data/quran_raw.csv", index=False, encoding="utf-8")
print("Data saved successfully to your disk!")
"""
    ),
    code(
        """
# You can also load it locally after saving:
# df = pd.read_csv("data/quran_raw.csv")
# print(f"Loaded {len(df)} rows. Ready for processing.")
"""
    ),
    code("df.shape"),
    code("df.head()"),
    code(
        """
if "df_load" in locals():
    del df_load
gc.collect()
"""
    ),
    md("## 3️⃣ Exploratory Data Analysis (EDA)"),
    md("### Information"),
    code("df.info()"),
    md("### Check Columns"),
    code("df.columns"),
    md("### Description"),
    code("df.describe(include='all')"),
    md("### Check Duplications"),
    code(
        """
print("Duplicated rows:", df.duplicated().sum())
print("Duplicated global ayah numbers:", df.duplicated("ayah_no_quran").sum())
"""
    ),
    code(
        """
df.drop_duplicates(subset=["ayah_no_quran"], keep="first", inplace=True)
df.reset_index(drop=True, inplace=True)
df.shape
"""
    ),
    md("### Check Missing Values"),
    code("df.isna().sum()"),
    md("### Show Number of Unique Values"),
    code(
        """
for col in df.columns:
    print("{} : {} unique value(s)".format(col, df[col].nunique()))
"""
    ),
    md("### Show Most Commen in Columns"),
    code(
        """
top_surahs = df["surah_name"].value_counts().head(15).reset_index()
top_surahs.columns = ["Surah", "Number of Ayahs"]
top_surahs
"""
    ),
    code(
        """
all_words = " ".join(df["text"].astype(str)).split()
word_counts = Counter(all_words)

top_words = pd.DataFrame(word_counts.most_common(15), columns=["Word", "Frequency"])
top_words
"""
    ),
    md("### Make Columns Length of Sentence"),
    code(
        """
df["number_of_characters_ayah"] = df["text"].astype(str).str.len()
df["number_of_words_ayah"] = df["text"].astype(str).str.split().str.len()
"""
    ),
    code(
        """
print(f"Maximum number of characters in an ayah is: {df['number_of_characters_ayah'].max()}")
print(f"Minimum number of characters in an ayah is: {df['number_of_characters_ayah'].min()}")
print(f"Average number of characters in an ayah is: {df['number_of_characters_ayah'].mean():.2f}")
"""
    ),
    md("#### Distribution of Length Characters"),
    code(
        """
plt.figure(figsize=(10, 5))
sns.histplot(df["number_of_characters_ayah"], bins=50, kde=True, color="steelblue")
plt.title("Distribution of Ayah Length by Characters")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()
"""
    ),
    md("#### Distribution of Length Words"),
    code(
        """
plt.figure(figsize=(10, 5))
sns.histplot(df["number_of_words_ayah"], bins=40, kde=True, color="tomato")
plt.title("Distribution of Ayah Length by Words")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
"""
    ),
    md("#### Generating Word Cloud For Sentences"),
    code(
        """
# A bar chart is used instead of a word cloud to avoid extra dependencies.
plt.figure(figsize=(10, 5))
sns.barplot(data=top_words, x="Frequency", y="Word", color="seagreen")
plt.title("Most Common Tokens in Quran Ayahs")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()
"""
    ),
    md("## 4️⃣ Processing"),
    md("### Clean Texts"),
    code(
        """
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text
"""
    ),
    md("### Arabic Normalization"),
    code(
        """
ARABIC_DIACRITICS_RE = re.compile(
    r"[\\u0610-\\u061A\\u064B-\\u065F\\u0670\\u06D6-\\u06ED]"
)


def normalize_arabic(text):
    text = str(text)
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[^\\u0600-\\u06FF\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text
"""
    ),
    md("### Apply Functions"),
    code(
        """
df["text"] = df["text"].apply(clean_text)
df["clean_text"] = df["text"].apply(normalize_arabic)
"""
    ),
    code("df[[\"text\", \"clean_text\"]].head()"),
    md("### Tokenization"),
    code(
        """
# In this retrieval project, tokenization is handled internally by the
# SentenceTransformer model. The output is a dense vector embedding.

model_checkpoint = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_checkpoint)

print("Model loaded:", model_checkpoint)
"""
    ),
    code(
        """
sample_texts = df["clean_text"].head(3).tolist()
sample_embeddings = model.encode(sample_texts, normalize_embeddings=True)

print("Sample embeddings shape:", sample_embeddings.shape)
"""
    ),
    md("### Convert DF To Dataset"),
    code(
        """
retrieval_dataset = df[
    [
        "surah_no",
        "surah_name",
        "ayah_number",
        "ayah_no_quran",
        "text",
        "clean_text",
    ]
].copy()

retrieval_dataset.head()
"""
    ),
    code(
        """
os.makedirs("data", exist_ok=True)
retrieval_dataset.to_csv("data/quran.csv", index=False, encoding="utf-8")
print("Processed retrieval dataset saved successfully!")
"""
    ),
    md("### Spliting Data"),
    code(
        """
# The Quran text itself is used as the retrieval corpus.
# For evaluation, we create a small query test set with expected ayahs.

test_cases = [
    {"query": "الحمد لله رب العلمين", "surah_name": "الفاتحة", "ayah_number": 2},
    {"query": "قل هو الله احد", "surah_name": "الإخلاص", "ayah_number": 1},
    {"query": "صم بكم عمي فهم", "surah_name": "البقرة", "ayah_number": 18},
    {"query": "مالك يوم الدين", "surah_name": "الفاتحة", "ayah_number": 4},
    {"query": "اياك نعبد واياك نستعين", "surah_name": "الفاتحة", "ayah_number": 5},
    {"query": "من الجنة والناس", "surah_name": "الناس", "ayah_number": 6},
]

test_df = pd.DataFrame(test_cases)
test_df
"""
    ),
    md("## 5️⃣ Modeling"),
    md("### Preparing"),
    code(
        """
ayah_texts = retrieval_dataset["clean_text"].tolist()
"""
    ),
    code(
        """
def score_text_match(clean_input, clean_text):
    if clean_text.startswith(clean_input):
        return 100.0
    if clean_input in clean_text:
        return 96.0
    return max(
        fuzz.WRatio(clean_input, clean_text),
        fuzz.partial_ratio(clean_input, clean_text),
        fuzz.token_set_ratio(clean_input, clean_text),
    )


def match_priority(clean_input, clean_text):
    if clean_text.startswith(clean_input):
        return 2
    if clean_input in clean_text:
        return 1
    return 0
"""
    ),
    code(
        """
def build_audio_url(ayah_no_quran, edition="ar.alafasy"):
    return f"https://cdn.islamic.network/quran/audio/128/{edition}/{int(ayah_no_quran)}.mp3"
"""
    ),
    md("### Train Data"),
    code(
        """
# There is no supervised training step.
# We build an embedding index for all Quran ayahs.

ayah_embeddings = model.encode(
    ayah_texts,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True,
)

ayah_embeddings.shape
"""
    ),
    code(
        """
def find_closest_ayah(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    if not clean_input:
        return pd.DataFrame()

    query_embedding = model.encode([clean_input], normalize_embeddings=True)
    semantic_scores = (query_embedding @ ayah_embeddings.T)[0]

    semantic_top_indices = np.argsort(semantic_scores)[::-1][: max(top_k * 8, 40)]

    fuzzy_scores = retrieval_dataset["clean_text"].apply(
        lambda text: score_text_match(clean_input, text)
    ).to_numpy()
    fuzzy_top_indices = np.argsort(fuzzy_scores)[::-1][: max(top_k * 8, 40)]

    candidate_indices = np.array(
        sorted(set(semantic_top_indices.tolist()) | set(fuzzy_top_indices.tolist()))
    )

    candidates = retrieval_dataset.iloc[candidate_indices].copy()
    candidates["semantic_score"] = semantic_scores[candidate_indices]
    candidates["fuzzy_score"] = fuzzy_scores[candidate_indices]
    candidates["match_priority"] = candidates["clean_text"].apply(
        lambda text: match_priority(clean_input, text)
    )
    candidates["final_score"] = (
        candidates["semantic_score"] * 100 * 0.35 + candidates["fuzzy_score"] * 0.65
    )

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
            "semantic_score",
            "fuzzy_score",
            "final_score",
        ]
    ]
"""
    ),
    md("### Evaluation"),
    code(
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
                "query": case["query"],
                "expected": f'{case["surah_name"]}:{case["ayah_number"]}',
                "top_prediction": f'{top_prediction["surah_name"]}:{top_prediction["ayah_number"]}',
                "rank": rank,
                "top1_correct": bool(rank == 1),
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
                "Top-1 Accuracy",
                f"Recall@{top_k}",
                "MRR",
                "Mean Rank",
                f"nDCG@{top_k}",
            ],
            "Score": [
                details["top1_correct"].mean(),
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
    code(
        """
evaluation_details, evaluation_summary = evaluate_retrieval(test_cases, top_k=5)
evaluation_details
"""
    ),
    code("evaluation_summary"),
    code(
        """
plot_df = evaluation_summary[evaluation_summary["Metric"] != "Mean Rank"].copy()

plt.figure(figsize=(9, 4))
sns.barplot(data=plot_df, x="Metric", y="Score", color="mediumpurple")
plt.ylim(0, 1.05)
plt.title("Retrieval Evaluation Metrics")
plt.xticks(rotation=20)
plt.show()
"""
    ),
    code(
        """
def find_embedding_only(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    query_embedding = model.encode([clean_input], normalize_embeddings=True)
    semantic_scores = (query_embedding @ ayah_embeddings.T)[0]
    top_indices = np.argsort(semantic_scores)[::-1][:top_k]

    candidates = retrieval_dataset.iloc[top_indices].copy()
    candidates["final_score"] = semantic_scores[top_indices] * 100
    return candidates


def find_fuzzy_only(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    fuzzy_scores = retrieval_dataset["clean_text"].apply(
        lambda text: score_text_match(clean_input, text)
    ).to_numpy()

    candidates = retrieval_dataset.copy()
    candidates["fuzzy_score"] = fuzzy_scores
    candidates["match_priority"] = candidates["clean_text"].apply(
        lambda text: match_priority(clean_input, text)
    )
    candidates["final_score"] = candidates["fuzzy_score"]
    return candidates.sort_values(["match_priority", "final_score"], ascending=False).head(top_k)
"""
    ),
    code(
        """
def evaluate_method(method_name, search_function, test_cases, top_k=5):
    rows = []

    for case in test_cases:
        predictions = search_function(case["query"], top_k=top_k).reset_index(drop=True)
        expected_mask = (
            (predictions["surah_name"] == case["surah_name"])
            & (predictions["ayah_number"] == case["ayah_number"])
        ).to_numpy()

        rank = int(np.where(expected_mask)[0][0]) + 1 if expected_mask.any() else None
        rows.append(
            {
                "Method": method_name,
                "Top-1 Accuracy": int(rank == 1),
                f"Recall@{top_k}": int(rank is not None),
                "MRR": 1 / rank if rank else 0,
                f"nDCG@{top_k}": 1 / np.log2(rank + 1) if rank else 0,
            }
        )

    return pd.DataFrame(rows).groupby("Method", as_index=False).mean()
"""
    ),
    code(
        """
comparison_summary = pd.concat(
    [
        evaluate_method("Embedding Only", find_embedding_only, test_cases),
        evaluate_method("Fuzzy Only", find_fuzzy_only, test_cases),
        evaluate_method("Hybrid System", find_closest_ayah, test_cases),
    ],
    ignore_index=True,
)

comparison_summary
"""
    ),
    code(
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
    md(
        """
#### **📊 Quran Ayah Retrieval Evaluation Metrics Guide**

In this project, we evaluate the system as a retrieval model rather than a normal classifier.

| Metric | Meaning |
| :--- | :--- |
| **Top-1 Accuracy** | The correct ayah is the first returned result |
| **Recall@5** | The correct ayah appears in the top 5 results |
| **MRR** | Rewards the model when the correct ayah appears at a higher rank |
| **nDCG@5** | Measures ranking quality in the top 5 results |

For Quran ayah correction, ranking metrics such as **MRR** and **nDCG@5** are more informative than simple accuracy.
"""
    ),
    md("## 6️⃣ Testing"),
    code(
        """
test_sentence = "الحمد لله رب العلمين"
print(find_closest_ayah(test_sentence, top_k=3)[["surah_name", "ayah_number", "text", "final_score"]])
"""
    ),
    code(
        """
test_sentence = "قل هو الله احد"
print(find_closest_ayah(test_sentence, top_k=3)[["surah_name", "ayah_number", "text", "final_score"]])
"""
    ),
    code(
        """
test_sentence = "صم بكم عمي فهم"
print(find_closest_ayah(test_sentence, top_k=3)[["surah_name", "ayah_number", "text", "final_score"]])
"""
    ),
    code(
        """
result = find_closest_ayah("الحمد لله رب العلمين", top_k=1).iloc[0]
audio_url = build_audio_url(result["ayah_no_quran"])

print(result["text"])
print(audio_url)
display(Audio(audio_url))
"""
    ),
    code(
        """
# Streamlit chatbot app:
# streamlit run app.py
#
# Open:
# http://localhost:8501
"""
    ),
    md("## **Thank You**🎀💌💓"),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
print(f"Cells: {len(cells)}")
