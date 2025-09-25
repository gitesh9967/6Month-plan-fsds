# XML SCRAPING FOR SML SHEET
import os
import xml.etree.ElementTree as ET
import re
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Change working directory
os.chdir(r"C:\Users\rosha\OneDrive\Desktop\Gitesh\NIT\NLP\WEB-SCRAPING\xml_single articles")

# Parse XML
tree = ET.parse("769952.xml")
root = tree.getroot()

# Convert XML tree to string
root = ET.tostring(root, encoding='utf8').decode('utf8')

# -------------------------
# Cleaning functions
# -------------------------
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub(r'\s+', ' ', text)   # replace multiple spaces/newlines with one
    return text.strip()

# -------------------------
# Example usage
# -------------------------
sample = denoise_text(root)
print(sample[:500])  # print first 500 chars to check