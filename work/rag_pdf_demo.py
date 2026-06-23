#!/usr/bin/env python3
"""
Minimal local PDF RAG demo with no API key.

What it does:
  - extracts text from a PDF
  - chunks the text
  - scores chunks against a question with TF-IDF
  - returns the best evidence snippets
  - builds a simple extractive answer from the top evidence

Usage:
  /path/to/python3 work/rag_pdf_demo.py /path/to/document.pdf "What does it say about refunds?"

Notes:
  - This is a local proof of concept, not a full generative LLM.
  - It is useful to validate whether your document can support RAG.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from pypdf import PdfReader


WORD_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aqui", "aquí", "como", "con",
    "cual", "cuál", "cuales", "cuáles", "de", "del", "desde", "donde", "dónde", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eres", "es", "esa", "esas", "ese", "eso", "esta",
    "está", "estas", "este", "esto", "estos", "fue", "ha", "hace", "han", "hasta", "hay", "la",
    "las", "le", "les", "lo", "los", "me", "mi", "mis", "mucho", "muy", "nada", "ni", "no",
    "nos", "nosotros", "o", "para", "pero", "por", "que", "qué", "se", "sin", "sobre", "su",
    "sus", "te", "tu", "tú", "un", "una", "uno", "unos", "y", "ya", "yo", "what", "which",
    "who", "whom", "when", "where", "why", "how", "the", "a", "an", "and", "or", "to", "of",
    "of", "of", "of", "in", "on", "for", "with", "of", "is", "are", "was", "were", "be",
}


@dataclass
class Chunk:
    page: int
    text: str


def extract_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = " ".join(text.split())
        pages.append(text)
    return pages


def chunk_pages(pages: List[str], chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, text in enumerate(pages, start=1):
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(page=page_num, text=chunk_text))
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def tokenize(text: str) -> List[str]:
    tokens = [token.lower() for token in WORD_RE.findall(normalize_text(text))]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def build_tfidf(chunks: List[Chunk]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    docs = [tokenize(chunk.text) for chunk in chunks]
    doc_freq: Counter[str] = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))

    n_docs = max(1, len(docs))
    idf: Dict[str, float] = {
        term: math.log((1 + n_docs) / (1 + df)) + 1.0
        for term, df in doc_freq.items()
    }

    tfidf_docs: List[Dict[str, float]] = []
    for tokens in docs:
        counts = Counter(tokens)
        total = max(1, len(tokens))
        weights: Dict[str, float] = {}
        for term, count in counts.items():
            weights[term] = (count / total) * idf.get(term, 0.0)
        tfidf_docs.append(weights)

    return tfidf_docs, idf


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = max(1, len(tokens))
    return {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, weight in a.items():
        dot += weight * b.get(term, 0.0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_chunks(question: str, chunks: List[Chunk], tfidf_docs: List[Dict[str, float]], idf: Dict[str, float], top_k: int = 4) -> List[Tuple[int, float]]:
    q_tokens = tokenize(question)
    q_vec = tfidf_vector(q_tokens, idf)
    scored = []
    q_norm = normalize_text(question)
    for idx, doc_vec in enumerate(tfidf_docs):
        tfidf_score = cosine_sparse(q_vec, doc_vec)
        chunk_tokens = set(tokenize(chunks[idx].text))
        overlap_score = len(set(q_tokens) & chunk_tokens) / max(1, len(set(q_tokens)))
        phrase_hits = 0.0
        for token in q_tokens:
            if token in normalize_text(chunks[idx].text):
                phrase_hits += 0.15
        if q_norm and q_norm in normalize_text(chunks[idx].text):
            phrase_hits += 0.5
        score = tfidf_score + 0.6 * overlap_score + phrase_hits
        scored.append((idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def best_sentences(text: str, question: str, max_sentences: int = 2) -> List[str]:
    question_terms = set(tokenize(question))
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return [text.strip()]

    scored: List[Tuple[float, str]] = []
    for sentence in sentences:
        sentence_terms = set(tokenize(sentence))
        overlap = len(question_terms & sentence_terms)
        length_penalty = max(1.0, math.log(len(sentence_terms) + 1))
        normalized_sentence = normalize_text(sentence)
        phrase_bonus = 0.0
        for token in question_terms:
            if token in normalized_sentence:
                phrase_bonus += 0.2
        score = (overlap / length_penalty) + phrase_bonus
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = [sentence for score, sentence in scored[:max_sentences] if score > 0]
    return chosen or [sentences[0]]


def format_context(chunks: List[Chunk], ranked: List[Tuple[int, float]]) -> str:
    parts = []
    for i, (idx, score) in enumerate(ranked, start=1):
        chunk = chunks[idx]
        parts.append(f"[{i}] Page {chunk.page} | score={score:.3f}\n{chunk.text}")
    return "\n\n".join(parts)


def build_answer(question: str, chunks: List[Chunk], ranked: List[Tuple[int, float]]) -> str:
    if not ranked:
        return "No encontré evidencia suficiente."

    top_chunks = [chunks[idx] for idx, _ in ranked[:2]]
    evidence_sentences: List[str] = []
    for chunk in top_chunks:
        evidence_sentences.extend(best_sentences(chunk.text, question, max_sentences=1))

    deduped: List[str] = []
    for sentence in evidence_sentences:
        if sentence not in deduped:
            deduped.append(sentence)

    if not deduped:
        return "No encontré evidencia suficiente."

    citations = ", ".join(f"[{i}]" for i in range(1, min(3, len(ranked)) + 1))
    answer = " ".join(deduped[:2]).strip()
    confidence = ranked[0][1]
    if confidence < 0.10:
        return f"{answer}\n\nCitas: {citations}\nConfianza baja: el texto parece relacionado, pero la coincidencia es débil."
    return f"{answer}\n\nCitas: {citations}"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 work/rag_pdf_demo.py /path/to/document.pdf 'Your question'")
        print("       python3 work/rag_pdf_demo.py /path/to/document.pdf")
        return 1

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return 1

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        print("No text found in the PDF.")
        return 1

    tfidf_docs, idf = build_tfidf(chunks)

    def run_question(question: str) -> None:
        ranked = rank_chunks(question, chunks, tfidf_docs, idf, top_k=4)
        print("\n=== Question ===")
        print(question)
        print("\n=== Retrieved Context ===")
        print(format_context(chunks, ranked))
        print("\n=== Local Answer ===")
        print(build_answer(question, chunks, ranked))

    if len(sys.argv) > 2:
        question = " ".join(sys.argv[2:]).strip()
        run_question(question)
        return 0

    print("Interactive mode. Press Enter on an empty line to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            break
        if not question:
            break
        run_question(question)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
