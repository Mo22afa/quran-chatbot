import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from preprocessing import normalize_arabic


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class QuranMatcher:
    def __init__(self, csv_path, model_name=None):
        self.csv_path = csv_path
        self.model_name = model_name or os.getenv("QURAN_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.df = self._load_quran_csv(csv_path)
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = self._build_embeddings()

    def _load_quran_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={
                "surah": "surah_name",
                "sura_name": "surah_name",
                "sura": "surah_name",
                "surah_name_ar": "surah_name",
                "ayah": "ayah_number",
                "aya": "ayah_number",
                "verse": "ayah_number",
                "aya_number": "ayah_number",
                "ayah_no_surah": "ayah_number",
                "verse_text": "text",
                "ayah_text": "text",
                "aya_text": "text",
                "ayah_ar": "text",
            }
        )

        required = {"surah_name", "ayah_number", "text"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                "quran.csv must contain these columns: surah_name, ayah_number, text. "
                f"Missing: {', '.join(missing)}"
            )

        df = df.dropna(subset=["text"]).copy()
        if "ayah_no_quran" in df.columns:
            df = df.drop_duplicates(subset=["ayah_no_quran"], keep="first")
        else:
            df = df.drop_duplicates(subset=["surah_name", "ayah_number", "text"], keep="first")
        df["clean_text"] = df["text"].apply(normalize_arabic)
        return df.reset_index(drop=True)

    def _format_for_model(self, texts, prefix):
        if "e5" in self.model_name.lower():
            return [f"{prefix}: {text}" for text in texts]
        return list(texts)

    def _build_embeddings(self):
        passages = self._format_for_model(self.df["clean_text"].tolist(), "passage")
        return self.model.encode(
            passages,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    def search(self, user_input, top_k=5):
        clean_input = normalize_arabic(user_input)
        if not clean_input:
            return []

        query = self._format_for_model([clean_input], "query")
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        semantic_scores = (query_embedding @ self.embeddings.T)[0]
        semantic_top_indices = np.argsort(semantic_scores)[::-1][: max(top_k * 8, 40)]

        fuzzy_scores = self.df["clean_text"].apply(
            lambda text: self._score_text_match(clean_input, text)
        ).to_numpy()
        fuzzy_top_indices = np.argsort(fuzzy_scores)[::-1][: max(top_k * 8, 40)]

        candidate_indices = np.array(
            sorted(set(semantic_top_indices.tolist()) | set(fuzzy_top_indices.tolist()))
        )

        candidates = self.df.iloc[candidate_indices].copy()
        candidates["semantic_score"] = semantic_scores[candidate_indices]
        candidates["fuzzy_score"] = fuzzy_scores[candidate_indices]
        candidates["match_priority"] = candidates["clean_text"].apply(
            lambda text: self._match_priority(clean_input, text)
        )
        candidates["final_score"] = (
            candidates["semantic_score"] * 100 * 0.35 + candidates["fuzzy_score"] * 0.65
        )

        candidates = candidates.sort_values(
            ["match_priority", "final_score"], ascending=False
        ).head(top_k)

        output_columns = [
            "surah_name",
            "ayah_number",
            "text",
            "semantic_score",
            "fuzzy_score",
            "final_score",
        ]
        for optional_column in ["surah_no", "ayah_no_quran"]:
            if optional_column in candidates.columns:
                output_columns.insert(0, optional_column)

        return candidates[output_columns].to_dict("records")

    def prefix_search(self, user_input, top_k=10):
        clean_input = normalize_arabic(user_input)
        if len(clean_input) < 2:
            return []

        candidates = self.df.copy()
        candidates["prefix_score"] = candidates["clean_text"].apply(
            lambda text: self._score_prefix_match(clean_input, text)
        )

        if candidates["prefix_score"].max() < 80:
            candidates["prefix_score"] = candidates["clean_text"].apply(
                lambda text: self._score_fuzzy_prefix_match(clean_input, text)
            )

        candidates = candidates[candidates["prefix_score"] > 0].copy()

        if candidates.empty:
            return []

        candidates["final_score"] = candidates["prefix_score"]
        candidates = candidates.sort_values(
            ["prefix_score", "ayah_no_quran"] if "ayah_no_quran" in candidates.columns else ["prefix_score"],
            ascending=[False, True] if "ayah_no_quran" in candidates.columns else [False],
        ).head(top_k)

        output_columns = [
            "surah_name",
            "ayah_number",
            "text",
            "final_score",
        ]
        for optional_column in ["surah_no", "ayah_no_quran"]:
            if optional_column in candidates.columns:
                output_columns.insert(0, optional_column)

        return candidates[output_columns].to_dict("records")

    def _score_text_match(self, clean_input, clean_text):
        if clean_text.startswith(clean_input):
            return 100.0
        if clean_input in clean_text:
            return 96.0
        return max(
            fuzz.WRatio(clean_input, clean_text),
            fuzz.partial_ratio(clean_input, clean_text),
            fuzz.token_set_ratio(clean_input, clean_text),
        )

    def _match_priority(self, clean_input, clean_text):
        if clean_text.startswith(clean_input):
            return 2
        if clean_input in clean_text:
            return 1
        return 0

    def _score_prefix_match(self, clean_input, clean_text):
        if clean_text.startswith(clean_input):
            return 100.0

        words = clean_text.split()
        if any(word.startswith(clean_input) for word in words):
            return 92.0

        if clean_input in clean_text:
            return 85.0

        return 0.0

    def _score_fuzzy_prefix_match(self, clean_input, clean_text):
        clean_words = clean_text.split()
        input_words = clean_input.split()
        if not clean_words or not input_words:
            return 0.0

        prefix_window = " ".join(clean_words[: max(len(input_words), 4)])
        full_score = fuzz.partial_ratio(clean_input, clean_text)
        prefix_score = fuzz.partial_ratio(clean_input, prefix_window)
        token_score = fuzz.token_set_ratio(clean_input, clean_text)
        word_score = self._score_ordered_word_overlap(input_words, clean_words)

        score = max(full_score, prefix_score, token_score, word_score)
        if clean_text.startswith(input_words[0]):
            score += 5
        return min(float(score), 99.0) if score >= 65 else 0.0

    def _score_ordered_word_overlap(self, input_words, clean_words):
        matched = 0
        position = 0

        for input_word in input_words:
            if len(input_word) <= 1:
                continue

            for index in range(position, len(clean_words)):
                clean_word = clean_words[index]
                if (
                    input_word == clean_word
                    or clean_word.startswith(input_word)
                    or fuzz.ratio(input_word, clean_word) >= 80
                ):
                    matched += 1
                    position = index + 1
                    break

        if matched == 0:
            return 0.0

        coverage = matched / max(len(input_words), 1)
        if coverage >= 0.75:
            return 98.0
        if coverage >= 0.55:
            return 88.0
        if coverage >= 0.40:
            return 75.0
        return 0.0
