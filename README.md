# LLM-translate-and-log-to-csv
CLI tool (Python) to translate raw text (TXT) via an OpenAI model and log results to bilingual CSV for CAT import
You can adjust chunk size to produce smaller segments. The segmentation breaks at ends of sentences, but larger chunk size leads to multiple-sentence segments.

"""
translate_chunks_to_csv.py

Translate a TXT file in chunks using a system prompt from a separate file,
and write a bilingual CSV: one row per source chunk with its translation.

Outputs:
  - <base>_aligned.csv          (live, appended while running)
  - <base>_aligned_sorted.csv   (final, strictly sorted by chunk index)
  - <base>_translated.txt       (concatenated translation, for convenience)
  - <base>_backup.txt           (line-based backup: "index||translation")

Usage example in Python prompt:

#Set your API key:
export OPENAI_API_KEY=sk-...

#Run the script with parameters  
  python translate_chunks_to_csv.py \
      --input book.txt \
      --sysprompt-file sysprompt.txt \
      --model gpt-5 \
      --threads 4 \
      --chunk-size 6000
"""
