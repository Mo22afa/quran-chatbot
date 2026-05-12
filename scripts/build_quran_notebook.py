import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "Quran_Ayah_Correction_Chatbot.ipynb"


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# Project - Quran Ayah Correction Chatbot **(LAB)**

This notebook follows the same workflow style as a standard NLP project:

1. Required Libraries
2. Read Dataset
3. Exploratory Data Analysis (EDA)
4. Processing
5. Modeling
6. Evaluation
7. Testing
8. Chatbot App
"""
    ),
    md("## 1. Required Libraries"),
    code(
        """
# Basic Libraries
import os
import re
import gc
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP / Matching Libraries
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# Notebook Display
from IPython.display import Audio, display

pd.set_option("display.max_colwidth", 160)
plt.style.use("seaborn-v0_8")
"""
    ),
    md("## 2. Read Dataset"),
    md("### Read From GitHub"),
    code(
        """
DATA_URL = "https://raw.githubusercontent.com/malekverse/quran-dataset/main/quran_dataset.csv"

df_load = pd.read_csv(DATA_URL)
df_load.head()
"""
    ),
    md(
        """
#### Project Data Source: Quran Dataset

This project uses a complete Quran dataset hosted on GitHub. The dataset contains Arabic ayah text and useful metadata, including surah name, ayah number inside the surah, and global ayah number in the Quran.

The most important columns for this project are:

| Column | Description |
|---|---|
| `surah_no` | Surah number |
| `surah_name_ar` | Arabic surah name |
| `ayah_no_surah` | Ayah number inside the surah |
| `ayah_no_quran` | Global ayah number from 1 to 6236 |
| `ayah_ar` | Arabic ayah text |
| `ayah_en` | English translation |

The system uses the Arabic Quran text for retrieval and correction. The model does not generate Quranic text; it only retrieves the closest ayah from this dataset.
"""
    ),
    md("### Select and Rename Columns"),
    code(
        """
df = df_load.rename(
    columns={
        "surah_name_ar": "surah_name",
        "ayah_no_surah": "ayah_number",
        "ayah_ar": "text",
    }
)

selected_columns = [
    "surah_no",
    "surah_name",
    "ayah_number",
    "ayah_no_quran",
    "text",
    "ayah_en",
]

df = df[selected_columns].copy()
df.head()
"""
    ),
    code(
        """
# The original CSV contains duplicated ayah rows.
# We keep one row for each global ayah number.
print("Before removing duplicates:", df.shape)
df = df.drop_duplicates(subset=["ayah_no_quran"], keep="first").reset_index(drop=True)
print("After removing duplicates:", df.shape)
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
    md("## 3. Exploratory Data Analysis (EDA)"),
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
    md("### Check Missing Values"),
    code("df.isna().sum()"),
    md("### Show Number of Unique Values"),
    code(
        """
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique value(s)")
"""
    ),
    md("### Show Most Common Surahs by Number of Ayahs"),
    code(
        """
surah_counts = df["surah_name"].value_counts().reset_index()
surah_counts.columns = ["surah_name", "ayah_count"]
surah_counts.head(15)
"""
    ),
    code(
        """
plt.figure(figsize=(12, 5))
sns.barplot(data=surah_counts.head(15), x="ayah_count", y="surah_name", color="steelblue")
plt.title("Top 15 Surahs by Number of Ayahs")
plt.xlabel("Number of Ayahs")
plt.ylabel("Surah")
plt.show()
"""
    ),
    md("### Make Columns Length of Ayah"),
    code(
        """
df["number_of_characters"] = df["text"].astype(str).str.len()
df["number_of_words"] = df["text"].astype(str).str.split().str.len()

print("Maximum number of characters:", df["number_of_characters"].max())
print("Minimum number of characters:", df["number_of_characters"].min())
print("Average number of characters:", round(df["number_of_characters"].mean(), 2))

df[["text", "number_of_characters", "number_of_words"]].head()
"""
    ),
    md("#### Distribution of Length Characters"),
    code(
        """
plt.figure(figsize=(10, 5))
sns.histplot(df["number_of_characters"], bins=50, kde=True, color="teal")
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
sns.histplot(df["number_of_words"], bins=40, kde=True, color="tomato")
plt.title("Distribution of Ayah Length by Words")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
"""
    ),
    md("#### Most Common Arabic Tokens"),
    code(
        """
all_words = " ".join(df["text"].astype(str)).split()
word_counts = Counter(all_words)

common_words = pd.DataFrame(word_counts.most_common(20), columns=["word", "count"])
common_words
"""
    ),
    md("## 4. Processing"),
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
df["clean_text"] = df["text"].apply(normalize_arabic)
df[["text", "clean_text"]].head()
"""
    ),
    code(
        """
example = "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ"
print("Original:", example)
print("Normalized:", normalize_arabic(example))
"""
    ),
    md("### Save Processed Dataset"),
    code(
        """
os.makedirs("data", exist_ok=True)
df.to_csv("data/quran.csv", index=False, encoding="utf-8")
print("Processed dataset saved to data/quran.csv")
"""
    ),
    md("## 5. Modeling"),
    md("### Preparing Retrieval Model"),
    code(
        """
model_checkpoint = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_checkpoint)

print("Model loaded:", model_checkpoint)
"""
    ),
    md("### Generate Ayah Embeddings"),
    code(
        """
ayah_texts = df["clean_text"].tolist()

ayah_embeddings = model.encode(
    ayah_texts,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True,
)

ayah_embeddings.shape
"""
    ),
    md("### Matching Functions"),
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
    md("### Prediction Function"),
    code(
        """
def find_closest_ayah(user_input, top_k=5):
    clean_input = normalize_arabic(user_input)
    if not clean_input:
        return pd.DataFrame()

    query_embedding = model.encode([clean_input], normalize_embeddings=True)
    semantic_scores = (query_embedding @ ayah_embeddings.T)[0]

    semantic_top_indices = np.argsort(semantic_scores)[::-1][: max(top_k * 8, 40)]

    fuzzy_scores = df["clean_text"].apply(
        lambda text: score_text_match(clean_input, text)
    ).to_numpy()
    fuzzy_top_indices = np.argsort(fuzzy_scores)[::-1][: max(top_k * 8, 40)]

    candidate_indices = np.array(
        sorted(set(semantic_top_indices.tolist()) | set(fuzzy_top_indices.tolist()))
    )

    candidates = df.iloc[candidate_indices].copy()
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
    code(
        """
find_closest_ayah("الحمد لله رب العلمين", top_k=5)
"""
    ),
    md("## 6. Evaluation"),
    md("### Evaluation Dataset"),
    code(
        """
test_cases = [
    {"query": "الحمد لله رب العلمين", "surah_name": "الفاتحة", "ayah_number": 2},
    {"query": "قل هو الله احد", "surah_name": "الإخلاص", "ayah_number": 1},
    {"query": "صم بكم عمي فهم", "surah_name": "البقرة", "ayah_number": 18},
    {"query": "مالك يوم الدين", "surah_name": "الفاتحة", "ayah_number": 4},
    {"query": "اياك نعبد واياك نستعين", "surah_name": "الفاتحة", "ayah_number": 5},
    {"query": "من الجنة والناس", "surah_name": "الناس", "ayah_number": 6},
]

eval_df = pd.DataFrame(test_cases)
eval_df
"""
    ),
    md("### Calculate Top-K Accuracy"),
    code(
        """
evaluation_rows = []

for case in test_cases:
    predictions = find_closest_ayah(case["query"], top_k=5).reset_index(drop=True)

    top1 = predictions.iloc[0]
    expected_match = (
        (predictions["surah_name"] == case["surah_name"])
        & (predictions["ayah_number"] == case["ayah_number"])
    )

    top1_correct = bool(expected_match.iloc[0])
    top5_correct = bool(expected_match.any())

    rank = None
    if top5_correct:
        rank = int(np.where(expected_match.to_numpy())[0][0]) + 1

    evaluation_rows.append(
        {
            "query": case["query"],
            "expected": f'{case["surah_name"]}:{case["ayah_number"]}',
            "predicted": f'{top1["surah_name"]}:{top1["ayah_number"]}',
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "rank": rank,
            "score": round(float(top1["final_score"]), 2),
        }
    )

results_df = pd.DataFrame(evaluation_rows)
results_df
"""
    ),
    code(
        """
top1_accuracy = results_df["top1_correct"].mean()
top5_accuracy = results_df["top5_correct"].mean()
mrr = np.mean([1 / rank if rank else 0 for rank in results_df["rank"]])

metrics = pd.DataFrame(
    {
        "Metric": ["Top-1 Accuracy", "Top-5 Accuracy", "MRR"],
        "Score": [top1_accuracy, top5_accuracy, mrr],
    }
)

metrics
"""
    ),
    code(
        """
plt.figure(figsize=(7, 4))
sns.barplot(data=metrics, x="Metric", y="Score", color="seagreen")
plt.ylim(0, 1.05)
plt.title("Retrieval Evaluation Metrics")
plt.show()
"""
    ),
    md(
        """
#### Evaluation Metrics Guide

| Metric | Meaning |
|---|---|
| Top-1 Accuracy | The correct ayah is the first returned result |
| Top-5 Accuracy | The correct ayah exists in the top 5 returned results |
| MRR | Mean Reciprocal Rank, which rewards higher ranking of the correct ayah |

For this project, Top-1 Accuracy is the most important metric because the chatbot displays the first result as the main correction.
"""
    ),
    md("## 7. Audio Feature"),
    code(
        """
AUDIO_CDN_TEMPLATE = "https://cdn.islamic.network/quran/audio/128/{edition}/{ayah_no_quran}.mp3"
DEFAULT_AUDIO_EDITION = "ar.alafasy"


def build_audio_url(ayah_no_quran, edition=DEFAULT_AUDIO_EDITION):
    return AUDIO_CDN_TEMPLATE.format(
        edition=edition,
        ayah_no_quran=int(ayah_no_quran),
    )
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
    md("## 8. Testing"),
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
    md("## 9. Chatbot App"),
    code(
        """
# The production chatbot interface is implemented in app.py using Streamlit.
# Run this command in the terminal from the project folder:
#
# streamlit run app.py
#
# Then open:
# http://localhost:8501
"""
    ),
    md(
        """
## Conclusion

This notebook builds the Quran Ayah Correction Chatbot using the same NLP project workflow as the reference notebook. The final system combines Arabic preprocessing, Hugging Face embeddings, RapidFuzz matching, evaluation, testing, and a Streamlit chatbot app.

The model is used for retrieval only. It does not generate Quranic text.

## Thank You
"""
    ),
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


OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(OUTPUT_PATH)
