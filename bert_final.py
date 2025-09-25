#!/usr/bin/env python3
"""
BERTopic-based SEC Filings Pipeline:
 - Clean HTML / TXT SEC filings
 - Extract metadata (Company, CIK, Year)
 - Extract DEI and Financial text (body only, store titles separately)
 - Run BERTopic on DEI sentences, paragraphs, and financial sentences
 - Save to Excel with topic assignments
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import html
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

from bertopic import BERTopic
from umap import UMAP
import hdbscan

# ---------- CONFIG ----------
DIVERSITY_GLOSSARY_PATH = "DEI_Glossary.xlsx"
TOPIC_NUM = 8


# ---------- IO + Cleaning ----------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def clean_sec_text(raw_html: str) -> str:
    """Cleaner for SEC HTML/text files."""
    soup = BeautifulSoup(raw_html, "lxml")
    for t in soup.find_all(["table", "script", "style", "noscript", "iframe"]):
        t.decompose()
    for tag in soup.find_all():
        if tag.name and ":" in tag.name:
            tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\|\|+", " ", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", " ", text)
    text = re.sub(r"\b[A-Z]{2,5}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- Metadata ----------
def extract_from_raw_header(raw: str) -> Tuple[str, str, str]:
    company = ""
    cik = ""
    year = ""
    m_name = re.search(r"COMPANY CONFORMED NAME:\s*(.+?)(?:\r|\n)", raw, re.I)
    if m_name:
        company = m_name.group(1).strip()
    m_cik = re.search(r"CENTRAL INDEX KEY:\s*([0-9\-]{4,20})", raw, re.I)
    if m_cik:
        cik = m_cik.group(1).strip().lstrip("0")
    m_year = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})", raw, re.I)
    if m_year:
        year = m_year.group(1).strip()
    return company, cik, year


def extract_from_html_title(raw: str) -> Tuple[str, str]:
    soup = BeautifulSoup(raw, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if title:
        company_guess = title.split("-")[0].strip()
        return company_guess, title
    return "", title


def extract_from_filename(fname: str) -> Tuple[str, str, str]:
    stem = Path(fname).stem
    m_cik = re.search(r"(\d{7,10})", fname)
    cik = m_cik.group(1) if m_cik else ""
    m_year = re.search(r"(20\d{2}|19\d{2})", fname)
    year = m_year.group(1) if m_year else ""
    parts = re.split(r"[_\-\.]", stem)
    company = parts[0] if parts else ""
    return company, cik, year


def consolidate_metadata(filename: Path, raw: str) -> Tuple[str, str, str]:
    comp, cik, year = extract_from_raw_header(raw)
    if not comp or not cik or not year:
        title_comp, _ = extract_from_html_title(raw)
        if not comp and title_comp:
            comp = title_comp
    if not comp or not cik or not year:
        f_comp, f_cik, f_year = extract_from_filename(str(filename.name))
        if not comp and f_comp:
            comp = f_comp
        if not cik and f_cik:
            cik = f_cik
        if not year and f_year:
            year = f_year
    if (not comp or not cik) and filename.parent:
        parent = filename.parent.name
        p_comp, p_cik, p_year = extract_from_filename(parent)
        if not comp and p_comp:
            comp = p_comp
        if not cik and p_cik:
            cik = p_cik
        if not year and p_year:
            year = p_year
    cik = cik.strip()
    if cik.isdigit() and len(cik) < 10:
        cik = cik.zfill(10)
    return comp.strip(), cik, year


# ---------- Glossary ----------
def load_glossary(path: str) -> List[str]:
    df = pd.read_excel(path, engine="openpyxl")
    terms = []
    for col in df.columns:
        terms.extend(df[col].dropna().astype(str).tolist())
    cleaned = [re.sub(r"\s+", " ", t).strip().lower() for t in terms if t]
    return sorted(set(cleaned))


def compile_term_patterns(terms: List[str]):
    out = []
    for t in terms:
        t_flexible = re.escape(t).replace(r"\ ", r"\s+")
        pat = re.compile(r"\b" + t_flexible + r"\b", flags=re.I)
        out.append((t, pat))
    return out


# ---------- MD&A Extraction ----------
# def extract_mdna_text(raw_text: str) -> Tuple[str, str]:
#     """
#     Extract MD&A section body text while removing only the first heading line.
#     Returns (body_text, title_guess).
#     """
#     mdna_match = re.search(
#         r"(item\s*7[^a-zA-Z]{0,3}.*?management.*?discussion.*?analysis.*?)(?=item\s*7a|item\s*8|item\s*9|signatures?|$)",
#         raw_text, flags=re.I | re.S
#     )
#     if not mdna_match:
#         return "", ""

#     mdna_chunk = mdna_match.group(1).strip()
#     mdna_chunk_clean = BeautifulSoup(mdna_chunk, "lxml").get_text(" ")
#     mdna_chunk_clean = re.sub(r"\s+", " ", mdna_chunk_clean).strip()

#     lines = [ln.strip() for ln in re.split(r"[.\n\r]+", mdna_chunk_clean) if ln.strip()]
#     if not lines:
#         return "", ""

#     title_guess = lines[0]
#     body_lines = lines[1:]
#     body_text = " ".join(body_lines).strip()
#     return body_text, title_guess

# def extract_mdna_text(raw_text: str):
#     mdna_patterns = [
#         r"(item\s*7\.*\s*management.*?discussion.*?analysis.*?)(?=item\s*7a|item\s*8|signatures?|$)",
#         r"(management.?discussion.?analysis.*?)(?=item\s*7a|item\s*8|signatures?|$)",
#         r"(mda\s*[-:].*?)(?=item\s*7a|item\s*8|signatures?|$)"
#     ]
    
#     for pat in mdna_patterns:
#         match = re.search(pat, raw_text, flags=re.I | re.S)
#         if match:
#             chunk = match.group(1)
#             text = BeautifulSoup(chunk, "lxml").get_text(" ")
#             text = re.sub(r"\s+", " ", text).strip()
#             lines = [ln.strip() for ln in re.split(r"[.\n\r]+", text)]
#             if len(lines) > 1:
#                 title = lines[0]
#                 body = " ".join([ln for ln in lines[1:] if len(ln.split()) > 5])
#                 return body, title
#     return "", ""

def extract_main_document(raw_text: str) -> str:
    """
    Try to isolate the main <DOCUMENT> or <TEXT> block for the filing (prefer 10-K).
    Fallback to entire raw_text if none found.
    """
    # 1) If EDGAR-style <DOCUMENT> blocks exist, prefer the DOCUMENT whose <TYPE> contains '10-K'
    docs = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw_text, flags=re.I | re.S)
    if docs:
        for d in docs:
            # find TYPE inside doc
            m = re.search(r"<TYPE>(.*?)\n", d, flags=re.I)
            if m and "10-k" in m.group(1).lower():
                return d
        # fallback: return the largest document chunk (often the primary)
        largest = max(docs, key=len)
        return largest

    # 2) Try <TEXT> blocks
    texts = re.findall(r"<TEXT>(.*?)</TEXT>", raw_text, flags=re.I | re.S)
    if texts:
        # same logic: pick the biggest text block
        return max(texts, key=len)

    # 3) fallback: return the whole raw text
    return raw_text


def is_valid_mdna_sentence(s: str, min_words: int = 6) -> bool:
    """Heuristics to filter out headings, TOC, or HTML-remnant noise."""
    if not s or not s.strip():
        return False
    s = s.strip()

    # filter HTML remnants or tags
    if re.search(r"<\/?[a-z][\s\S]*?>", s):
        return False

    # must contain some letters
    if len(re.findall(r"[A-Za-z]", s)) < 4:
        return False

    # skip short sentences
    if len(s.split()) < min_words:
        return False

    # skip TOC-like lines that contain many 'Item' repetitions (e.g., index lines)
    if len(re.findall(r"\bItem\b", s, flags=re.I)) >= 2:
        return False

    # skip lines that are mostly uppercase (likely headings)
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            return False

    # skip lines with weird punctuation-only content
    if re.sub(r"[A-Za-z0-9]", "", s).strip() and len(re.sub(r"[A-Za-z0-9\s]", "", s)) / max(1, len(s)) > 0.5:
        return False

    # skip lines that are clearly a location/address/table-of-contents
    if re.search(r"(location|address|quarter\s+ended|high \$|exhibit|item\s+\[reserved\])", s, flags=re.I):
        return False

    return True

# def extract_mdna_text(raw_text: str):
#     """
#     Extract MD&A section with multiple passes:
#     1. Check tables first
#     2. Try Item 7 → fallback to Item 7A → fallback to Item 8
#     3. Preserve title even if body is empty
#     4. Log empty cases for QA
#     """
#     # --- Step 1: Extract table text before removing tables ---
#     soup = BeautifulSoup(raw_text, "lxml")
#     table_text = " ".join([tbl.get_text(" ", strip=True) for tbl in soup.find_all("table")])
#     if table_text:
#         table_text = re.sub(r"\s+", " ", table_text)

#     # --- Step 2: Search patterns with fallback ---
#     mdna_patterns = [
#         r"(item\s*7\.*\s*management.*?discussion.*?analysis.*?)(?=item\s*7a|item\s*8|signatures?|$)",
#         r"(management.?discussion.?analysis.*?)(?=item\s*7a|item\s*8|signatures?|$)",
#         r"(mda\s*[-:].*?)(?=item\s*7a|item\s*8|signatures?|$)"
#     ]
    
#     mdna_chunk, title_guess = "", ""
#     for pat in mdna_patterns:
#         match = re.search(pat, raw_text, flags=re.I | re.S)
#         if match:
#             mdna_chunk = match.group(1)
#             break

#     # --- Step 3: Clean up chunk or fallback to tables ---
#     if mdna_chunk:
#         text = BeautifulSoup(mdna_chunk, "lxml").get_text(" ")
#         text = re.sub(r"\s+", " ", text).strip()
#     else:
#         # Fallback: use table text if no main body found
#         text = table_text

#     # --- Step 4: Split title & body ---
#     if text:
#         lines = [ln.strip() for ln in re.split(r"[.\n\r]+", text) if ln.strip()]
#         if len(lines) > 1:
#             title_guess = lines[0]
#             body = " ".join([ln for ln in lines[1:] if len(ln.split()) > 5])
#         else:
#             title_guess = lines[0]
#             body = ""  # Only title present
#     else:
#         body, title_guess = "", ""

#     # --- Step 5: Log empty bodies for QA ---
#     if not body:
#         print(f"[INFO] MD&A body empty. Title='{title_guess[:50]}'")

#     return body, title_guess

def extract_mdna_text(raw_text: str) -> Tuple[str, str]:
    """
    Robust MD&A extractor:
     - isolate main DOCUMENT / TEXT if present
     - search for Item 7 heading but avoid TOC matches
     - fallback: find 'Management' phrase directly and grab trailing text
     - fallback: extract table text if present
    Returns (body_text, title_guess)
    """
    # 1) operate on main document chunk first
    main_doc = extract_main_document(raw_text)
    search_space = main_doc

    # 2) pre-extract table text (we'll use as a fallback)
    soup_full = BeautifulSoup(main_doc, "lxml")
    table_text = " ".join([tbl.get_text(" ", strip=True) for tbl in soup_full.find_all("table")])
    if table_text:
        table_text = re.sub(r"\s+", " ", table_text).strip()

    # 3) Try strong Item 7 heading (line-based and avoiding TOC hits)
    # require 'Item 7' on its own line or followed by punctuation/newline and 'management' phrase
    pat_item7 = re.compile(
        r"(?:^|\n)\s*(item\s*7[\.\:\)]?\s*(?:[:\-\.\)]\s*)?[^.\n]{0,200}?management[^.\n]{0,200}?discussion[^.\n]{0,200}?analysis.*?)"
        r"(?=\n(?:item\s*7a|item\s*8|item\s*9|signatures?|$))",
        flags=re.I | re.S
    )
    m = pat_item7.search(search_space)
    mdna_chunk = ""
    if m:
        candidate = m.group(1).strip()
        # guard: if the candidate looks like a TOC (contains several 'Item X' occurrences), skip it
        if len(re.findall(r"\bItem\b", candidate, flags=re.I)) <= 1:
            mdna_chunk = candidate

    # 4) fallback: look for explicit 'Management's Discussion' phrase and grab a window after it
    if not mdna_chunk:
        m2 = re.search(r"(management[’'s]*\s+discussion(?:\s+and\s+analysis)?)(.*?)(?=(item\s*7a|item\s*8|item\s*9|signatures?|$))",
                       search_space, flags=re.I | re.S)
        if m2:
            mdna_chunk = (m2.group(1) + " " + (m2.group(2) or "")).strip()

    # 5) final fallback: if no chunk found, use table text
    if not mdna_chunk and table_text:
        mdna_chunk = table_text

    # 6) if still nothing -> return empty (caller can log)
    if not mdna_chunk:
        return "", ""

    # 7) Clean the candidate chunk and split into lines, drop TOC-like short lines
    chunk_text = BeautifulSoup(mdna_chunk, "lxml").get_text(" ")
    chunk_text = html.unescape(chunk_text)
    chunk_text = re.sub(r"\s+", " ", chunk_text).strip()

    # break on sentence delimiters/newlines but keep longer sentences intact
    # we split by sentence-ending punctuation and newlines to find title and body
    lines = [ln.strip() for ln in re.split(r"[\r\n]+|\.\s{1,}|\.\s*$", chunk_text) if ln.strip()]

    # remove lines that are obviously TOC-like or very short headings
    cleaned_lines = [ln for ln in lines if ln and not (len(re.findall(r"\bItem\b", ln, flags=re.I)) > 1 or len(ln.split()) < 4)]

    if not cleaned_lines:
        # keep original first non-empty line as title if nothing else
        title_guess = lines[0] if lines else ""
        return "", title_guess

    # pick first cleaned line containing 'management' as title if present, else first cleaned line
    title_guess = ""
    for ln in cleaned_lines:
        if re.search(r"management", ln, flags=re.I):
            title_guess = ln
            break
    if not title_guess:
        title_guess = cleaned_lines[0]

    # Make body: take subsequent cleaned lines that pass validity filter
    body_lines = []
    start_collect = False
    for ln in cleaned_lines:
        if not start_collect:
            if ln == title_guess or re.search(r"management", ln, flags=re.I):
                start_collect = True
            continue
        if is_valid_mdna_sentence(ln):
            body_lines.append(ln)

    # If none collected, try a looser approach: find sentences in chunk_text after the title_guess
    if not body_lines:
        trailing = chunk_text.split(title_guess, 1)[-1] if title_guess in chunk_text else chunk_text
        # split into sentences and pick those that pass basic filters
        for s in re.split(r"(?<=[\.\?!])\s+", trailing):
            s = s.strip()
            if is_valid_mdna_sentence(s):
                body_lines.append(s)
        # if still empty, leave empty body (we will log and keep title)
    body_text = " ".join(body_lines).strip()
    return body_text, title_guess

# ---------- BERTopic Runner ----------
# def run_bertopic(sentences: List[str]):
#     if not sentences:
#         return None

#     n_points = len(sentences)
#     if n_points == 1:
#         return {"topic_ids": [0], "topic_labels": {0: "SingleSentence"}}
#     if n_points == 2:
#         return {"topic_ids": [0, 1], "topic_labels": {0: "Topic0", 1: "Topic1"}}

#     n_neighbors = max(2, min(5, n_points - 1))
#     min_cluster_size = min(2, n_points)
#     min_samples = min(1, n_points)

#     umap_model = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.1, random_state=42)
#     hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)

#     model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)
#     topics, _ = model.fit_transform(sentences)
#     topic_info = model.get_topic_info()

#     topic_map = {row["Topic"]: " ".join([w for w, _ in model.get_topic(row["Topic"])])
#                  for _, row in topic_info.iterrows() if row["Topic"] != -1}
#     topic_labels = {tid: topic_map.get(tid, "Misc") for tid in set(topics)}

#     return {"topic_ids": topics, "topic_labels": topic_labels}

def run_bertopic(sentences: List[str]):
    if not sentences:
        return None

    n_points = len(sentences)

    # Case 1: Very few sentences → direct topic assignment
    if n_points == 1:
        return {"topic_ids": [0], "topic_labels": {0: "SingleSentence"}}
    if n_points == 2:
        return {"topic_ids": [0, 1], "topic_labels": {0: "Topic0", 1: "Topic1"}}

    # Dynamically reduce n_neighbors so it's < n_points
    n_neighbors = max(2, min(5, n_points - 1))
    min_cluster_size = min(2, n_points)
    min_samples = min(1, n_points)

    try:
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=min(2, n_points - 1),
            min_dist=0.1,
            random_state=42
        )

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=False
        )

        topics, _ = model.fit_transform(sentences)
        topic_info = model.get_topic_info()

        topic_map = {
            row["Topic"]: " ".join([w for w, _ in model.get_topic(row["Topic"])])
            for _, row in topic_info.iterrows() if row["Topic"] != -1
        }
        topic_labels = {tid: topic_map.get(tid, "Misc") for tid in set(topics)}

        return {"topic_ids": topics, "topic_labels": topic_labels}

    except Exception as e:
        print(f"[WARN] BERTopic failed on small corpus: {e}")
        # Fallback: assign everything to one topic
        return {"topic_ids": [0] * n_points, "topic_labels": {0: "AllContent"}}

# ---------- Pipeline ----------
# def run_pipeline(src_dir: str, out_xlsx: str, limit: int = 100):
#     src = Path(src_dir)
#     if not src.exists():
#         raise FileNotFoundError(src)

#     div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
#     div_term_patterns = compile_term_patterns(div_terms)

#     files = sorted(list(src.glob("*.txt")) + list(src.glob("*.htm")) + list(src.glob("*.html")))
#     files = files[:limit]

#     dei_sentence_texts, dei_paragraph_texts, fin_sentence_texts = [], [], []
#     dei_sent_meta, dei_para_meta, fin_sent_meta = [], [], []

#     for path in tqdm(files, desc="Scanning files"):
#         raw_for_meta = path.read_text(encoding="utf-8", errors="ignore")
#         cleaned = clean_sec_text(raw_for_meta)
#         company, cik, year = consolidate_metadata(path, raw_for_meta)
#         filing_type = "DEF14" if "def" in path.stem.lower() else "10-K"

#         # DEI Section
#         paragraphs = [p for p in re.split(r"(?:\r\n\r\n|\n\s*\n|\r\n|\n){1,}", cleaned) if p.strip()]
#         for p in paragraphs:
#             para_terms = [t for t, pat in div_term_patterns if pat.search(p)]
#             if para_terms:
#                 dei_paragraph_texts.append(p)
#                 para_index = len(dei_paragraph_texts) - 1
#                 dei_para_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
#                                       "filingtype": filing_type, "paragraph_text": p, "para_index": para_index})

#                 for s in sent_tokenize(p):
#                     if any(pat.search(s) for _, pat in div_term_patterns):
#                         dei_sentence_texts.append(s)
#                         sent_index = len(dei_sentence_texts) - 1
#                         matched = sorted(set([t for t, pat in div_term_patterns if pat.search(s)]))
#                         dei_sent_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
#                                               "filingtype": filing_type, "paragraph_text": p, "sentence_text": s,
#                                               "matched_terms": matched, "para_index": para_index,
#                                               "sent_index": sent_index})

#         # MD&A Section
#         # if filing_type == "10-K":
#         #     mdna_body, mdna_title = extract_mdna_text(raw_for_meta)
#         #     if mdna_body:
#         #         mdna_clean = clean_sec_text(mdna_body)
#         #         for s in sent_tokenize(mdna_clean):
#         #             if len(s.split()) > 5 and re.search(r"[A-Za-z]", s):
#         #                 fin_sentence_texts.append(s)
#         #                 fin_sent_meta.append({"company": company, "cik": cik, "year": year, "filingtype": filing_type,
#         #                                       "sentence_text": s, "file": path.name, "mdna_title": mdna_title,
#         #                                       "sent_index": len(fin_sentence_texts) - 1})

#         # MD&A Section
#         if filing_type == "10-K":
#             mdna_body, mdna_title = extract_mdna_text(raw_for_meta)
#             if mdna_body:
#                 mdna_clean = clean_sec_text(mdna_body)
#                 for s in sent_tokenize(mdna_clean):
#                     if len(s.split()) > 5 and re.search(r"[A-Za-z]", s):
#                         fin_sentence_texts.append(s)
#                         fin_sent_meta.append({
#                             "company": company, "cik": cik, "year": year,
#                             "filingtype": filing_type, "sentence_text": s,
#                             "file": path.name, "mdna_title": mdna_title,
#                             "sent_index": len(fin_sentence_texts) - 1
#                         })
#             else:
#                 # Log case with title but no body
#                 fin_sentence_texts.append("")  # Keeps row for QA
#                 fin_sent_meta.append({
#                     "company": company, "cik": cik, "year": year,
#                     "filingtype": filing_type, "sentence_text": "",
#                     "file": path.name, "mdna_title": mdna_title,
#                     "sent_index": len(fin_sentence_texts) - 1
#                 })


#     # Topic Modeling
#     div_sent_topics = run_bertopic(dei_sentence_texts)
#     div_para_topics = run_bertopic(dei_paragraph_texts)
#     fin_sent_topics = run_bertopic(fin_sentence_texts)

#     # Outputs
#     diversity_rows = []
#     sent_topic_ids = div_sent_topics["topic_ids"] if div_sent_topics else []
#     para_topic_ids = div_para_topics["topic_ids"] if div_para_topics else []

#     for meta in dei_sent_meta:
#         s_idx = meta["sent_index"]
#         p_idx = meta["para_index"]
#         tid_sent = sent_topic_ids[s_idx] if s_idx < len(sent_topic_ids) else None
#         label_sent = div_sent_topics["topic_labels"].get(tid_sent, "") if div_sent_topics else ""
#         tid_para = para_topic_ids[p_idx] if p_idx < len(para_topic_ids) else None
#         label_para = div_para_topics["topic_labels"].get(tid_para, "") if div_para_topics else ""
#         for word in meta["matched_terms"]:
#             diversity_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
#                                    "FilingType": meta["filingtype"], "DEI_Word": word,
#                                    "Sentence": meta["sentence_text"], "Paragraph": meta["paragraph_text"],
#                                    "Topic_ID_Sentence": tid_sent, "Topic_Label_Sentence": label_sent,
#                                    "Topic_ID_Paragraph": tid_para, "Topic_Label_Paragraph": label_para,
#                                    "SourceFile": meta["file"]})

#     financial_rows = []
#     fin_topic_ids = fin_sent_topics["topic_ids"] if fin_sent_topics else []
#     for meta in fin_sent_meta:
#         idx = meta["sent_index"]
#         tid = fin_topic_ids[idx] if idx < len(fin_topic_ids) else None
#         label = fin_sent_topics["topic_labels"].get(tid, "") if fin_sent_topics else ""
#         financial_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
#                                "FilingType": meta["filingtype"], "Sentence": meta["sentence_text"],
#                                "Topic_ID": tid, "Topic_Label": label, "SourceFile": meta["file"],
#                                "MDNA_Title": meta.get("mdna_title", "")})

#     # Save Excel
#     with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
#         pd.DataFrame(diversity_rows).to_excel(writer, sheet_name="diversity_output", index=False)
#         pd.DataFrame(financial_rows).to_excel(writer, sheet_name="financial_output", index=False)

#     print(f"Wrote {out_xlsx}")
#     print(f"DEI rows: {len(diversity_rows)}, Financial rows: {len(financial_rows)}")

# # ---------- CLI ----------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="Run BERTopic pipeline on SEC filings")
#     ap.add_argument("src_dir", help="Directory with filings (.txt/.htm/.html)")
#     ap.add_argument("out_xlsx", help="Output Excel filename")
#     ap.add_argument("--limit", type=int, default=200, help="Max number of files to process")
#     args = ap.parse_args()
#     run_pipeline(args.src_dir, args.out_xlsx, limit=args.limit)

# ---------- Pipeline ----------

# def run_pipeline(src_dirs: List[str], out_xlsx: str, limit: int = 100):
#     # Collect all files from multiple folders
#     all_files = []
#     for folder in src_dirs:
#         src = Path(folder)
#         if not src.exists():
#             raise FileNotFoundError(f"Folder not found: {folder}")
#         folder_files = sorted(list(src.glob("*.txt")) + list(src.glob("*.htm")) + list(src.glob("*.html")))
#         all_files.extend(folder_files)

#     all_files = all_files[:limit]

#     div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
#     div_term_patterns = compile_term_patterns(div_terms)

#     dei_sentence_texts, dei_paragraph_texts, fin_sentence_texts = [], [], []
#     dei_sent_meta, dei_para_meta, fin_sent_meta = [], [], []

#     for path in tqdm(all_files, desc="Scanning files"):
#         raw_for_meta = path.read_text(encoding="utf-8", errors="ignore")
#         cleaned = clean_sec_text(raw_for_meta)
#         company, cik, year = consolidate_metadata(path, raw_for_meta)
#         filing_type = "DEF14" if "def" in path.stem.lower() else "10-K"

#         # DEI Section
#         paragraphs = [p for p in re.split(r"(?:\r\n\r\n|\n\s*\n|\r\n|\n){1,}", cleaned) if p.strip()]
#         for p in paragraphs:
#             para_terms = [t for t, pat in div_term_patterns if pat.search(p)]
#             if para_terms:
#                 dei_paragraph_texts.append(p)
#                 para_index = len(dei_paragraph_texts) - 1
#                 dei_para_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
#                                       "filingtype": filing_type, "paragraph_text": p, "para_index": para_index})

#                 for s in sent_tokenize(p):
#                     if any(pat.search(s) for _, pat in div_term_patterns):
#                         dei_sentence_texts.append(s)
#                         sent_index = len(dei_sentence_texts) - 1
#                         matched = sorted(set([t for t, pat in div_term_patterns if pat.search(s)]))
#                         dei_sent_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
#                                               "filingtype": filing_type, "paragraph_text": p, "sentence_text": s,
#                                               "matched_terms": matched, "para_index": para_index,
#                                               "sent_index": sent_index})

#         # MD&A Section
#         if filing_type == "10-K":
#             mdna_body, mdna_title = extract_mdna_text(raw_for_meta)
#             if mdna_body:
#                 # clean and split
#                 mdna_clean = BeautifulSoup(mdna_body, "lxml").get_text(" ")
#                 mdna_clean = re.sub(r"\s+", " ", mdna_clean).strip()
#                 for s in sent_tokenize(mdna_clean):
#                     s = s.strip()
#                     if is_valid_mdna_sentence(s):
#                         fin_sentence_texts.append(s)
#                         fin_sent_meta.append({
#                             "company": company,
#                             "cik": cik,
#                             "year": year,
#                             "filingtype": filing_type,
#                             "sentence_text": s,
#                             "file": path.name,
#                             "mdna_title": mdna_title,
#                             "sent_index": len(fin_sentence_texts) - 1
#                         })
#             else:
#                 # Log missing body for QA (do NOT append empty sentence to corpus)
#                 print(f"[INFO] MD&A body empty for file: {path.name}; title='{mdna_title[:60]}'")
#                 # Optionally, keep a separate record so QA can review which files had no MD&A body:
#                 mdna_clean.append({"file": path.name, "company": company, "cik": cik, "year": year, "mdna_title": mdna_title})
            
#             # if mdna_body:
#             #     mdna_clean = clean_sec_text(mdna_body)
#             #     for s in sent_tokenize(mdna_clean):
#             #         if len(s.split()) > 5 and re.search(r"[A-Za-z]", s):
#             #             fin_sentence_texts.append(s)
#             #             fin_sent_meta.append({
#             #                 "company": company, "cik": cik, "year": year,
#             #                 "filingtype": filing_type, "sentence_text": s,
#             #                 "file": path.name, "mdna_title": mdna_title,
#             #                 "sent_index": len(fin_sentence_texts) - 1
#             #             })
#             # else:
#             #     fin_sentence_texts.append("")
#             #     fin_sent_meta.append({
#             #         "company": company, "cik": cik, "year": year,
#             #         "filingtype": filing_type, "sentence_text": "",
#             #         "file": path.name, "mdna_title": mdna_title,
#             #         "sent_index": len(fin_sentence_texts) - 1
#             #     })

#     # Topic Modeling
#     div_sent_topics = run_bertopic(dei_sentence_texts)
#     div_para_topics = run_bertopic(dei_paragraph_texts)
#     fin_sent_topics = run_bertopic(fin_sentence_texts)

#     # Outputs
#     diversity_rows = []
#     sent_topic_ids = div_sent_topics["topic_ids"] if div_sent_topics else []
#     para_topic_ids = div_para_topics["topic_ids"] if div_para_topics else []

#     for meta in dei_sent_meta:
#         s_idx = meta["sent_index"]
#         p_idx = meta["para_index"]
#         tid_sent = sent_topic_ids[s_idx] if s_idx < len(sent_topic_ids) else None
#         label_sent = div_sent_topics["topic_labels"].get(tid_sent, "") if div_sent_topics else ""
#         tid_para = para_topic_ids[p_idx] if p_idx < len(para_topic_ids) else None
#         label_para = div_para_topics["topic_labels"].get(tid_para, "") if div_para_topics else ""
#         for word in meta["matched_terms"]:
#             diversity_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
#                                    "FilingType": meta["filingtype"], "DEI_Word": word,
#                                    "Sentence": meta["sentence_text"], "Paragraph": meta["paragraph_text"],
#                                    "Topic_ID_Sentence": tid_sent, "Topic_Label_Sentence": label_sent,
#                                    "Topic_ID_Paragraph": tid_para, "Topic_Label_Paragraph": label_para,
#                                    "SourceFile": meta["file"]})

#     financial_rows = []
#     fin_topic_ids = fin_sent_topics["topic_ids"] if fin_sent_topics else []
#     for meta in fin_sent_meta:
#         idx = meta["sent_index"]
#         tid = fin_topic_ids[idx] if idx < len(fin_topic_ids) else None
#         label = fin_sent_topics["topic_labels"].get(tid, "") if fin_sent_topics else ""
#         financial_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
#                                "FilingType": meta["filingtype"], "Sentence": meta["sentence_text"],
#                                "Topic_ID": tid, "Topic_Label": label, "SourceFile": meta["file"],
#                                "MDNA_Title": meta.get("mdna_title", "")})

#     # Save Excel
#     with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
#         pd.DataFrame(diversity_rows).to_excel(writer, sheet_name="diversity_output", index=False)
#         pd.DataFrame(financial_rows).to_excel(writer, sheet_name="financial_output", index=False)

#     print(f"Wrote {out_xlsx}")
#     print(f"DEI rows: {len(diversity_rows)}, Financial rows: {len(financial_rows)}")

def run_pipeline(src_dirs: List[str], out_xlsx: str, limit: int = 100):
    # Collect all files from multiple folders
    all_files = []
    for folder in src_dirs:
        src = Path(folder)
        if not src.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        folder_files = sorted(list(src.glob("*.txt")) + list(src.glob("*.htm")) + list(src.glob("*.html")))
        all_files.extend(folder_files)

    all_files = all_files[:limit]

    div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
    div_term_patterns = compile_term_patterns(div_terms)

    dei_sentence_texts, dei_paragraph_texts, fin_sentence_texts = [], [], []
    dei_sent_meta, dei_para_meta, fin_sent_meta = [], [], []
    mdna_missing = []  # Track missing MD&A bodies for QA

    for path in tqdm(all_files, desc="Scanning files"):
        raw_for_meta = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_sec_text(raw_for_meta)
        company, cik, year = consolidate_metadata(path, raw_for_meta)
        filing_type = "DEF14" if "def" in path.stem.lower() else "10-K"

        # --- DEI Section ---
        paragraphs = [p for p in re.split(r"(?:\r\n\r\n|\n\s*\n|\r\n|\n){1,}", cleaned) if p.strip()]
        for p in paragraphs:
            para_terms = [t for t, pat in div_term_patterns if pat.search(p)]
            if para_terms:
                dei_paragraph_texts.append(p)
                para_index = len(dei_paragraph_texts) - 1
                dei_para_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
                                      "filingtype": filing_type, "paragraph_text": p, "para_index": para_index})

                for s in sent_tokenize(p):
                    if any(pat.search(s) for _, pat in div_term_patterns):
                        dei_sentence_texts.append(s)
                        sent_index = len(dei_sentence_texts) - 1
                        matched = sorted(set([t for t, pat in div_term_patterns if pat.search(s)]))
                        dei_sent_meta.append({"file": path.name, "company": company, "cik": cik, "year": year,
                                              "filingtype": filing_type, "paragraph_text": p, "sentence_text": s,
                                              "matched_terms": matched, "para_index": para_index,
                                              "sent_index": sent_index})

        # --- MD&A Section ---
        if filing_type == "10-K":
            mdna_body, mdna_title = extract_mdna_text(raw_for_meta)
            if mdna_body:
                mdna_clean = BeautifulSoup(mdna_body, "lxml").get_text(" ")
                mdna_clean = re.sub(r"\s+", " ", mdna_clean).strip()
                for s in sent_tokenize(mdna_clean):
                    s = s.strip()
                    if is_valid_mdna_sentence(s):  # Strong filter
                        fin_sentence_texts.append(s)
                        fin_sent_meta.append({
                            "company": company, "cik": cik, "year": year,
                            "filingtype": filing_type, "sentence_text": s,
                            "file": path.name, "mdna_title": mdna_title,
                            "sent_index": len(fin_sentence_texts) - 1
                        })
            else:
                print(f"[INFO] MD&A body empty for file: {path.name}; title='{mdna_title[:60]}'")
                mdna_missing.append({"file": path.name, "company": company, "cik": cik,
                                     "year": year, "mdna_title": mdna_title})

    # --- Topic Modeling ---
    div_sent_topics = run_bertopic(dei_sentence_texts)
    div_para_topics = run_bertopic(dei_paragraph_texts)
    fin_sent_topics = run_bertopic(fin_sentence_texts)

    # --- Outputs: Diversity ---
    diversity_rows = []
    sent_topic_ids = div_sent_topics["topic_ids"] if div_sent_topics else []
    para_topic_ids = div_para_topics["topic_ids"] if div_para_topics else []

    for meta in dei_sent_meta:
        s_idx = meta["sent_index"]
        p_idx = meta["para_index"]
        tid_sent = sent_topic_ids[s_idx] if s_idx < len(sent_topic_ids) else None
        label_sent = div_sent_topics["topic_labels"].get(tid_sent, "") if div_sent_topics else ""
        tid_para = para_topic_ids[p_idx] if p_idx < len(para_topic_ids) else None
        label_para = div_para_topics["topic_labels"].get(tid_para, "") if div_para_topics else ""
        for word in meta["matched_terms"]:
            diversity_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
                                   "FilingType": meta["filingtype"], "DEI_Word": word,
                                   "Sentence": meta["sentence_text"], "Paragraph": meta["paragraph_text"],
                                   "Topic_ID_Sentence": tid_sent, "Topic_Label_Sentence": label_sent,
                                   "Topic_ID_Paragraph": tid_para, "Topic_Label_Paragraph": label_para,
                                   "SourceFile": meta["file"]})

    # --- Outputs: Financial ---
    financial_rows = []
    fin_topic_ids = fin_sent_topics["topic_ids"] if fin_sent_topics else []
    for meta in fin_sent_meta:
        idx = meta["sent_index"]
        tid = fin_topic_ids[idx] if idx < len(fin_topic_ids) else None
        label = fin_sent_topics["topic_labels"].get(tid, "") if fin_sent_topics else ""
        financial_rows.append({"Company": meta["company"], "CIK": meta["cik"], "Year": meta["year"],
                               "FilingType": meta["filingtype"], "Sentence": meta["sentence_text"],
                               "Topic_ID": tid, "Topic_Label": label, "SourceFile": meta["file"],
                               "MDNA_Title": meta.get("mdna_title", "")})

    # --- Save Excel ---
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        pd.DataFrame(diversity_rows).to_excel(writer, sheet_name="diversity_output", index=False)
        pd.DataFrame(financial_rows).to_excel(writer, sheet_name="financial_output", index=False)
        if mdna_missing:  # Optional sheet for QA transparency
            pd.DataFrame(mdna_missing).to_excel(writer, sheet_name="mdna_missing", index=False)

    print(f"Wrote {out_xlsx}")
    print(f"DEI rows: {len(diversity_rows)}, Financial rows: {len(financial_rows)}, Missing MD&A: {len(mdna_missing)}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BERTopic pipeline on SEC filings")
    ap.add_argument("inputs", nargs="+", help="Multiple input folders followed by output Excel filename")
    ap.add_argument("--limit", type=int, default=200, help="Max number of files to process")
    args = ap.parse_args()

    # Last argument is the output file, all others are input folders
    *input_folders, out_file = args.inputs

    # Run pipeline on all folders
    run_pipeline(input_folders, out_file, limit=args.limit)

