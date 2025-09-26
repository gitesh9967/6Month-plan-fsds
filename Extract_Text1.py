# XML Scraping + Cleaning + Basic NLP + Wordcloud + Extractive Summary
# Filename: xml_scraping_sml.py
# Purpose: parse an XML article file, extract and clean text, perform basic NLP (NLTK + spaCy),
#          create a wordcloud, POS-tagging, and a simple extractive summary using sentence scoring.

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import logging

from bs4 import BeautifulSoup

# --- Optional NLP libraries ---
# NLTK for tokenization / POS tagging
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# WordCloud and plotting
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# spaCy for more advanced tokenization and sentence segmentation
import spacy

# heapq for selecting top sentences
from heapq import nlargest

# ---------------------------
# Configuration
# ---------------------------
INPUT_DIR = r"C:\Users\rosha\OneDrive\Desktop\Gitesh\NIT\NLP\WEB-SCRAPING\0.text ectract(html)\xml_single articles"
FILENAME = "769952.xml"
OUTPUT_CLEAN_TEXT = "cleaned_769952.txt"
WORDCLOUD_PNG = "wordcloud_769952.png"
SUMMARY_TXT = "summary_769952.txt"

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------
# Helper / cleaning functions
# ---------------------------

def normalize_unicode(text: str) -> str:
    """Normalize unicode (NFKC) to collapse composed/decomposed chars."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def strip_html(text: str) -> str:
    """Remove any HTML markup and return plain text. Keeps whitespace between elements."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def remove_between_square_brackets(text: str) -> str:
    """Remove content like [citation needed], [1], etc."""
    return re.sub(r"\[[^\]]*\]", "", text)


def collapse_whitespace(text: str) -> str:
    """Replace multiple whitespace (including newlines & tabs) with a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def denoise_text(raw: str) -> str:
    """Full denoising pipeline: HTML removal, square bracket removal, unicode normalization & whitespace cleanup."""
    if not raw:
        return ""
    text = strip_html(raw)
    text = remove_between_square_brackets(text)
    text = normalize_unicode(text)
    text = collapse_whitespace(text)
    return text

# ---------------------------
# XML extraction
# ---------------------------

def extract_text_from_xml(path: str) -> str:
    """Parse XML and return concatenated text content preserving element order.

    This uses ElementTree; if your files are huge consider iterparse.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # ''.join(root.itertext()) will combine all text nodes in order
    raw_text = "".join(root.itertext())
    return raw_text

# ---------------------------
# NLTK setup helper
# ---------------------------

def ensure_nltk_resources():
    """Make sure common NLTK resources are available. Run this once if you see errors."""
    resources = ["punkt", "stopwords", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            logging.info(f"Downloading NLTK resource: {res}")
            nltk.download(res)

# ---------------------------
# spaCy setup helper
# ---------------------------

def load_spacy_model(name: str = "en_core_web_sm"):
    try:
        nlp = spacy.load(name)
    except OSError:
        logging.info(f"spaCy model {name} not found. Trying to download...")
        # Attempt to download the small model (requires internet / CLI). If this fails, instruct the user.
        from spacy.cli import download

        try:
            download(name)
            nlp = spacy.load(name)
        except Exception as e:
            logging.error(
                "Failed to download spaCy model automatically. Please run: python -m spacy download en_core_web_sm"
            )
            raise
    return nlp

# ---------------------------
# Wordcloud helper
# ---------------------------

def create_wordcloud(text: str, out_png: str = WORDCLOUD_PNG):
    wc = WordCloud(width=420, height=200, margin=2, background_color='black', colormap='Accent', mode='RGBA')
    wc.generate(text)
    wc.to_file(out_png)
    logging.info(f"Wordcloud written to: {out_png}")

    # Also display inline if running interactively
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.show()

# ---------------------------
# Simple extractive summarizer (frequency + sentence scoring)
# ---------------------------

def build_frequency_table(doc, stopwords_set):
    """Given a spaCy Doc, build normalized frequency table for token.text.lower() keys."""
    freqs = {}
    for token in doc:
        if token.is_alpha and not token.is_stop and token.text.lower() not in stopwords_set:
            key = token.text.lower()
            freqs[key] = freqs.get(key, 0) + 1

    if not freqs:
        return freqs

    max_freq = max(freqs.values())
    for k in freqs:
        freqs[k] = freqs[k] / max_freq
    return freqs


def score_sentences(doc, freq_table):
    """Score each sentence by summing normalized freq of its words."""
    sent_scores = {}
    for sent in doc.sents:
        score = 0
        for token in sent:
            key = token.text.lower()
            if key in freq_table:
                score += freq_table[key]
        # Only keep sentences with non-zero score
        if score > 0:
            sent_scores[sent] = score
    return sent_scores


def extract_summary(spacy_doc, ratio=0.4):
    """Return a list of sentence strings representing the summary.

    ratio = proportion of sentences to keep (0.0 - 1.0). Default 0.4 like your example.
    """
    stopwords_set = set(spacy.lang.en.stop_words.STOP_WORDS)
    freq_table = build_frequency_table(spacy_doc, stopwords_set)
    sent_scores = score_sentences(spacy_doc, freq_table)

    if not sent_scores:
        return []

    sentence_count = len(list(spacy_doc.sents))
    select_n = max(1, int(sentence_count * ratio))

    top_sentences = nlargest(select_n, sent_scores, key=sent_scores.get)

    # Return sentences in their original order
    top_sentences_sorted = sorted(top_sentences, key=lambda s: s.start)
    return [sent.text.strip() for sent in top_sentences_sorted]

# ---------------------------
# Main flow
# ---------------------------

def main():
    os.chdir(INPUT_DIR)
    xml_path = os.path.join(INPUT_DIR, FILENAME)

    logging.info(f"Parsing XML file: {xml_path}")
    raw_text = extract_text_from_xml(xml_path)
    logging.info(f"Extracted raw text length: {len(raw_text)} characters")

    cleaned = denoise_text(raw_text)
    logging.info(f"Cleaned text length: {len(cleaned)} characters")

    # write cleaned text to file for inspection
    with open(OUTPUT_CLEAN_TEXT, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    logging.info(f"Cleaned text written to: {OUTPUT_CLEAN_TEXT}")

    # Create wordcloud
    try:
        create_wordcloud(cleaned, WORDCLOUD_PNG)
    except Exception as e:
        logging.error(f"Failed to create wordcloud: {e}")

    # Ensure NLTK resources (tokenizers/pos tagger)
    try:
        ensure_nltk_resources()
    except Exception:
        logging.warning("NLTK resource download may have failed; some NLTK features may error out.")

    # NLTK tokenization and POS tagging (as in your snippet)
    try:
        tokens = word_tokenize(cleaned)
        logging.info(f"Number of NLTK tokens: {len(tokens)}")

        # POS tagging (NLTK)
        pos_tags = nltk.pos_tag(tokens)
        # keep a small sample printed to console to avoid huge output
        logging.info("POS tags sample (first 50):\n%s", pos_tags[:50])
    except Exception as e:
        logging.error(f"NLTK tokenization / tagging failed: {e}")

    # Load spaCy model
    try:
        nlp = load_spacy_model("en_core_web_sm")
    except Exception:
        logging.error("spaCy model not available. Please install via: python -m spacy download en_core_web_sm")
        return

    # Create spaCy doc for more robust sentence segmentation
    doc = nlp(cleaned)

    # Print token-level attributes similar to your snippet (but limited)
    logging.info("spaCy token sample (text:pos):")
    sample_tokens = list(doc)[:80]
    for t in sample_tokens:
        logging.info(f"{t.text} : {t.pos_} -> {t.lemma_}, dep={t.dep_}, tag={t.tag_}, shape={t.shape_}, is_alpha={t.is_alpha}, is_stop={t.is_stop}")

    # Build and save extractive summary
    summary_sentences = extract_summary(doc, ratio=0.4)
    if summary_sentences:
        summary_text = "\n".join(summary_sentences)
        with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logging.info(f"Summary written to: {SUMMARY_TXT}")
        logging.info("Summary (printed below):\n%s", summary_text)
    else:
        logging.info("No summary sentences could be extracted.")


if __name__ == "__main__":
    main()
