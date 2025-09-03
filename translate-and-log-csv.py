#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translate_chunks_to_csv.py

Translate a TXT file in chunks using a system prompt from a separate file,
and write a bilingual CSV: one row per source chunk with its translation.

Outputs:
  - <base>_aligned.csv          (live, appended while running)
  - <base>_aligned_sorted.csv   (final, strictly sorted by chunk index)
  - <base>_translated.txt       (concatenated translation, for convenience)
  - <base>_backup.txt           (line-based backup: "index||translation")

Usage example:
  export OPENAI_API_KEY=sk-...
  python translate_chunks_to_csv.py \
      --input book.txt \
      --sysprompt-file sysprompt.txt \
      --model gpt-5 \
      --threads 4 \
      --chunk-size 6000
"""

import os
import re
import csv
import sys
import time
import random
import queue
import threading
import argparse
from typing import Dict, List, Tuple, Optional

# OpenAI SDK (v1.x). Install with: pip install openai
try:
    import openai
except Exception:
    openai = None  # we'll error out if you try to run without it

# ---------------------------
# Threading state / locks
# ---------------------------
translated_texts: Dict[int, str] = {}
translated_texts_lock = threading.Lock()
csv_lock = threading.Lock()
print_lock = threading.Lock()

# Will be set in main()
translation_queue: "queue.Queue[Optional[Tuple[int, str]]]" = None  # type: ignore

# Output paths (set in main)
bilingual_csv = ""
bilingual_csv_sorted = ""
backup_file = ""
output_txt = ""

def test_openai_api_key(model: str = "gpt-4o-mini") -> bool:
    """
    Quick sanity check: tries a 1-token request to verify the API key works.
    Returns True if successful, False otherwise.
    """
    if openai is None:
        print("OpenAI SDK not installed. Run: pip install openai")
        return False
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=10000,
        )
        print(f"API key test successful with model '{model}'.")
        return True
    except Exception as e:
        print(f"API key test failed: {e}")
        return False

# ---------------------------
# Helpers
# ---------------------------

def log(msg: str):
    with print_lock:
        print(msg, flush=True)

def init_csv(path: str):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "src_chars", "tgt_chars", "source_chunk", "target_chunk"])

def append_csv_row(path: str, index: int, src_text: str, tgt_text: str):
    with csv_lock:
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow([index, len(src_text), len(tgt_text or ""), src_text, tgt_text or ""])

def save_backup(index: int, translation: str):
    with open(backup_file, "a", encoding="utf-8") as f:
        f.write(f"{index}||{translation}\n")

def read_text(path: str, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()

# ---------- Sentence segmentation (EN/DE heuristic, no dependencies) ----------

_ABBR = {
    # English
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "p.m.", "a.m.", "fig.",
    # German
    "z.b.", "u.a.", "bzw.", "usw.", "d.h.", "sog.", "ca.", "Nr.".lower(), "s.", "u.u.", "vgl."
}

def _looks_like_boundary(prev_token: str, next_char: str) -> bool:
    # Do not split if previous token is an abbreviation
    if prev_token.lower() in _ABBR:
        return False
    # Prefer split when next char starts a new sentence (capital letter, quote, digit, or newline)
    return bool(re.match(r'[\s\n]+[A-ZÄÖÜ0-9"“]', next_char))

def split_sentences(paragraph: str) -> list[str]:
    """
    Conservative rule-based splitter:
    - splits on ., !, ?, … optionally followed by closing quotes
    - skips common abbreviations
    - keeps the delimiter with the sentence
    """
    s = paragraph.strip()
    if not s:
        return []
    sentences = []
    start = 0
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ".!?…":
            # consume closing quotes if present
            j = i + 1
            while j < n and s[j] in ['"', "”", "’", "»", "›", "'", ")"]:
                j += 1
            # inspect previous token (last word + dot)
            prev = s[start:j].rstrip()
            # last word ending with a dot
            m = re.search(r'(\b\w+\.)\s*$', prev, flags=re.UNICODE)
            prev_token = m.group(1).lower() if m else ""
            # lookahead context
            next_context = s[j:j+2]  # next couple of chars
            if _looks_like_boundary(prev_token, next_context or " "):
                sentences.append(s[start:j].strip())
                # advance to next non-space
                k = j
                while k < n and s[k].isspace():
                    k += 1
                start = k
                i = k
                continue
        i += 1
    # tail
    if start < n:
        sentences.append(s[start:].strip())
    # merge tiny fragments that slipped through
    merged = []
    for sent in sentences:
        if merged and len(sent) < 3:
            merged[-1] += (" " if not merged[-1].endswith((" ", "\n")) else "") + sent
        else:
            merged.append(sent)
    return merged

def chunk_source_text(input_file: str, encoding: str, max_chunk_size: int = 1, segmentation: str = "sentence") -> list[tuple[int, str]]:
    """
    Build chunks up to max_chunk_size.
    - segmentation='paragraph': previous behavior (paragraph blocks)
    - segmentation='sentence' : packs whole sentences; preserves paragraph gaps as '\n\n'
    """
    text = read_text(input_file, encoding)
    paragraphs = re.split(r"\n{2,}", text)  # keep single \n inside paragraphs
    chunks: list[tuple[int, str]] = []

    buf_parts: list[str] = []
    size = 0
    index = 1

    for p_idx, para in enumerate(paragraphs):
        para = para.rstrip("\r\n")
        units = [para] if segmentation != "sentence" else split_sentences(para)

        # If sentence segmentation yields nothing (empty para), just represent as empty marker
        if segmentation == "sentence" and not units:
            # If there is content in buffer, insert paragraph break for separation
            if buf_parts and size + 2 <= max_chunk_size:
                buf_parts.append("")  # will render as a paragraph break when joining
                size += 2
            continue

        for u in units:
            u_text = u
            # separator between units in same paragraph (sentence mode) is a space;
            # between paragraphs we'll inject a double newline below.
            sep = (" " if segmentation == "sentence" else ("\n\n" if buf_parts else ""))
            add_len = len(u_text) + (1 if segmentation == "sentence" and buf_parts and buf_parts[-1] else 0)

            # predicted size if we add this unit
            predicted = size + (1 if segmentation == "sentence" and buf_parts and buf_parts[-1] else 0) + len(u_text)

            if predicted <= max_chunk_size or not buf_parts:
                # append with appropriate spacing
                if segmentation == "sentence":
                    if buf_parts and buf_parts[-1]:
                        buf_parts[-1] += " " + u_text
                    else:
                        buf_parts.append(u_text)
                else:
                    # paragraph mode: append as its own block
                    buf_parts.append(u_text) if buf_parts else buf_parts.append(u_text)
                size = predicted
            else:
                # flush current chunk
                chunk_text = "\n\n".join([part for part in buf_parts if part is not None])
                chunks.append((index, chunk_text))
                index += 1
                buf_parts = [u_text]
                size = len(u_text)

        # after finishing a paragraph, insert a paragraph separator
        if buf_parts is not None:
            # use a marker to force a blank line between paragraphs inside a chunk
            buf_parts.append("")  # will render as '\n\n' on join
            size += 2
            # avoid trailing growth beyond max; if too large, flush now
            if size > max_chunk_size and len(buf_parts) > 1:
                chunk_text = "\n\n".join([part for part in buf_parts[:-1] if part is not None])
                chunks.append((index, chunk_text))
                index += 1
                # start new buffer beginning with the paragraph break effect for the next para
                last = buf_parts[-1]
                buf_parts = [] if last == "" else [last]
                size = 0

    # finalize
    if buf_parts:
        # remove trailing empty separator
        while buf_parts and buf_parts[-1] == "":
            buf_parts.pop()
        chunk_text = "\n\n".join(buf_parts)
        if chunk_text:
            chunks.append((index, chunk_text))

    return chunks

# ---------------------------
# Translation
# ---------------------------

def translate_text(text: str, sysprompt: str, model: str) -> str:
    """
    Translate 'text' using OpenAI Chat Completions with a system prompt loaded from file.
    """
    if openai is None:
        raise RuntimeError("OpenAI SDK not available. Install `pip install openai`.")

    # Using the Chat Completions API (python-openai v1.x)
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": text},
        ],
    )
    out = resp.choices[0].message.content or ""
    return out.replace("\r\n", "\n")

# ---------------------------
# Worker
# ---------------------------

def worker(sysprompt: str, model: str, max_retries: int = 5, retry_delay: float = 2.0):
    """
    Pulls (index, chunk) from queue, translates, writes live CSV + backup, stores in memory.
    Retries on common transient errors (rate limit, timeouts).
    """
    while True:
        task = translation_queue.get()
        if task is None:
            translation_queue.task_done()
            break

        index, chunk = task
        try:
            for attempt in range(max_retries):
                try:
                    translated = translate_text(chunk, sysprompt, model)

                    # in-memory store for final sorted outputs
                    with translated_texts_lock:
                        translated_texts[index] = translated

                    # Split the chunk and translation into sentences
                    src_sentences = split_sentences(chunk)
                    tgt_sentences = split_sentences(translated)

                    # Ensure both lists have the same number of sentences
                    if len(src_sentences) != len(tgt_sentences):
                        log(f"[ERR] Mismatch in sentence count for chunk {index}")
                    else:
                        # Append each sentence pair to the CSV
                        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
                            append_csv_row(bilingual_csv, index, src_sentence, tgt_sentence)

                    # incremental backup
                    save_backup(index, translated)

                    log(f"[OK] Chunk {index} ({len(chunk)} chars)")
                    break

                except Exception as e:
                    msg = str(e).lower()
                    if any(t in msg for t in ["rate limit", "overloaded", "timeout", "temporar"]):
                        wait_s = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        log(f"[Retry] {index}: {e} → waiting {wait_s:.2f}s")
                        time.sleep(wait_s)
                        continue
                    else:
                        log(f"[ERR] {index}: {e} (non-retryable)")
                        break
        finally:
            translation_queue.task_done()

# ---------------------------
# Main
# ---------------------------

def main():
    global translation_queue, bilingual_csv, bilingual_csv_sorted, backup_file, output_txt

    ap = argparse.ArgumentParser(description="Translate TXT in chunks (sysprompt from file) and write bilingual CSV.")
    ap.add_argument("--input", required=True, help="Path to source TXT")
    ap.add_argument("--sysprompt-file", required=True, help="Path to system prompt TXT")
    ap.add_argument("--encoding", default="utf-8", help="Source encoding (default: utf-8)")
    ap.add_argument("--chunk-size", type=int, default=6000, help="Max chars per chunk (default: 6000)")
    ap.add_argument("--threads", type=int, default=4, help="Worker threads (default: 4)")
    ap.add_argument("--model", default="gpt-5", help="OpenAI model name (default: gpt-5)")
    ap.add_argument("--segmentation", choices=["paragraph", "sentence"], default="paragraph",
            help="Chunking strategy (default: paragraph). Use 'sentence' for sentence-boundary chunks.")

    args = ap.parse_args()

    # Sanity check the API key
    if not test_openai_api_key(args.model):
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.sysprompt_file):
        print(f"Sysprompt file not found: {args.sysprompt_file}", file=sys.stderr)
        sys.exit(2)

    sysprompt = read_text(args.sysprompt_file, "utf-8").strip()
    if not sysprompt:
        print("Sysprompt file is empty.", file=sys.stderr)
        sys.exit(2)

    base, _ = os.path.splitext(args.input)
    bilingual_csv = f"{base}_aligned.csv"
    bilingual_csv_sorted = f"{base}_aligned_sorted.csv"
    backup_file = f"{base}_backup.txt"
    output_txt = f"{base}_translated.txt"

    # Fresh outputs
    init_csv(bilingual_csv)
    if os.path.exists(backup_file):
        os.remove(backup_file)

    # Build chunks
    chunks = chunk_source_text(args.input, args.encoding, args.chunk_size, args.segmentation)

    if not chunks:
        print("No chunks produced (input may be empty).", file=sys.stderr)
        sys.exit(1)
    src_by_index = dict(chunks)
    log(f"Built {len(chunks)} chunk(s). Live CSV → {bilingual_csv}")

    # Queue & workers
    translation_queue = queue.Queue()
    workers: List[threading.Thread] = []
    for _ in range(max(1, args.threads)):
        t = threading.Thread(target=worker, args=(sysprompt, args.model), daemon=True)
        t.start()
        workers.append(t)

    # Enqueue tasks
    for task in chunks:
        translation_queue.put(task)

    # Wait for all tasks
    translation_queue.join()

    # Stop workers
    for _ in workers:
        translation_queue.put(None)
    for t in workers:
        t.join()

    # Write concatenated translated TXT (sorted)
    with open(output_txt, "w", encoding="utf-8") as f:
        for idx in sorted(src_by_index.keys()):
            f.write((translated_texts.get(idx, "") or "") + "\n\n")

    # Write sorted bilingual CSV
    with open(bilingual_csv_sorted, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "src_chars", "tgt_chars", "source_chunk", "target_chunk"])
        for idx in sorted(src_by_index.keys()):
            s = src_by_index[idx]
            t = translated_texts.get(idx, "") or ""
            w.writerow([idx, len(s), len(t), s, t])

    missing = [i for i in src_by_index if i not in translated_texts]
    print("\nDone.")
    print(f"Live CSV:   {bilingual_csv}")
    print(f"Sorted CSV: {bilingual_csv_sorted}")
    print(f"Translated: {output_txt}")
    if missing:
        print(f"WARNING: {len(missing)} chunk(s) missing translation: {missing[:10]}{'...' if len(missing)>10 else ''}")

if __name__ == "__main__":
    main()
