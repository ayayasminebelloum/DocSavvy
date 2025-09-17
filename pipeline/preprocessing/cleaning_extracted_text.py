# cleaning_extracted_text.py
# --------------------------------------------------
# Cleans raw OCR-extracted text for consistency, readability, and model input
# Includes basic cleaning, post-OCR corrections, and fast multilingual spell correction
# --------------------------------------------------

import re
import unicodedata
from symspellpy import SymSpell, Verbosity
import pkg_resources
import os
import spacy

# OCR Confusions to correct
CHAR_SUBSTITUTIONS = {
    '0': 'o', '1': 'l', '5': 's', '6': 'b', '8': 'B', '|': 'l', '¢': 'c',
    '@': 'a', '!': 'i', '\\': '/', '“': '"', '”': '"', '‘': "'", '’': "'"
}

# Common text noise patterns
NOISE_PATTERNS = [
    r'[–—]',       # en/em dash
    r'[_~]',       # special underline/tilde
    r'\n+',        # repeated line breaks
    r'\s{2,}',     # multiple spaces
    r'[^\x00-\x7F]+'  # non-ASCII (strip accents later instead)
]

# Common OCR spacing error patterns
SPACING_PATTERNS = [
    (r'([a-z])([A-Z][a-z])', r'\1 \2'),      # camelCase to spaced words
    (r'([a-zA-Z])(\d)', r'\1 \2'),           # letterNumber to letter number
    (r'(\d)([a-zA-Z])', r'\1 \2'),           # numberLetter to number letter
    (r'\.(\w)', r'. \1'),                    # period followed by word without space
    (r'([a-z])([A-Z])', r'\1 \2'),           # Fix missing spaces between words
    (r'([,;:])([a-zA-Z0-9])', r'\1 \2'),     # Missing space after punctuation
    (r'to([A-Z])', r'to \1'),                # Special case for "to" prefix
    (r'and([A-Z])', r'and \1'),              # Special case for "and" prefix
    (r'of([a-zA-Z])', r'of \1'),             # Special case for "of" prefix
    (r'in([A-Z])', r'in \1'),                # Special case for "in" prefix
    (r'([a-z])the([A-Z])', r'\1 the \2'),    # Fix for "the" without spaces
    (r'([a-z])at([a-zA-Z])', r'\1 at \2')    # Fix for "at" without spaces
]

# Initialize SymSpell once
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load dictionary for English, French, Spanish (pre-built or custom)
DICT_PATH = os.path.join(os.path.dirname(__file__), 'frequency_dictionary_combined.txt')
if os.path.exists(DICT_PATH):
    sym_spell.load_dictionary(DICT_PATH, term_index=0, count_index=1)
else:
    # Fallback to default English dictionary
    sym_spell.load_dictionary(pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"), 0, 1)

# Load multilingual spaCy model for NER
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "xx_ent_wiki_sm"])
    nlp = spacy.load("xx_ent_wiki_sm")

def fix_spacing_issues(text: str) -> str:
    """
    Fix common OCR spacing issues where words are merged together without spaces.
    This function handles cases like "toUSA" → "to USA" or "jointexploration" → "joint exploration"
    """
    # Apply all the spacing patterns
    for pattern, replacement in SPACING_PATTERNS:
        text = re.sub(pattern, replacement, text)
    
    # Handle more complex word merging using dictionary-based approach
    # Split text into tokens by spaces
    tokens = text.split()
    corrected_tokens = []
    
    for token in tokens:
        if len(token) > 12 and not token.isupper():  # Long tokens might be merged words
            # Try to find word boundaries in long tokens
            new_token = token
            # Insert spaces at potential word boundaries (lowercase to uppercase transitions)
            new_token = re.sub(r'([a-z])([A-Z])', r'\1 \2', new_token)
            # Look for common prefixes and suffixes in the middle of words
            common_parts = ['and', 'the', 'with', 'for', 'from', 'to', 'of', 'in', 'at', 'by', 'on']
            for part in common_parts:
                new_token = re.sub(f'([a-z])({part})([A-Z])', r'\1 \2 \3', new_token, flags=re.IGNORECASE)
            
            corrected_tokens.append(new_token)
        else:
            corrected_tokens.append(token)
    
    return ' '.join(corrected_tokens)

def clean_text(text: str) -> str:
    """
    Perform basic cleaning of OCR or raw text:
    - Remove non-printable characters
    - Normalize unicode
    - Remove excessive whitespace
    - Fix misread characters
    - Lowercase (optional for ML consistency)
    """
    text = unicodedata.normalize("NFKC", text)

    for wrong, right in CHAR_SUBSTITUTIONS.items():
        text = text.replace(wrong, right)

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text)

    text = re.sub(r'([\.,])\1+', r'\1', text)
    text = re.sub(r'[^\w\s\.,\-\:/]', '', text)

    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
                   
    # Apply spacing fixes for OCR errors
    text = fix_spacing_issues(text)

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def advanced_ocr_cleanup(text: str) -> str:
    """
    Perform more advanced cleanup on OCR text:
    - Fix section misalignment
    - Group common blocks
    - Clean up typical OCR artifacts (like broken lines)
    - Normalize list bullets or amounts
    - Apply token-level spelling correction (fast using SymSpell)
    - Preserve named entities detected via spaCy (multilingual)
    """
    # First apply basic cleaning and spacing fixes
    text = clean_text(text)
    
    lines = text.split('\n') if '\n' in text else text.split('.')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        line = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', line)
        line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
        line = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', line)
        line = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', line)

        if len(re.sub(r'[^a-zA-Z0-9]', '', line)) < 3:
            continue

        cleaned_lines.append(line)

    joined_text = ' '.join(cleaned_lines)
    joined_text = re.sub(r'\s{2,}', ' ', joined_text).strip()

    # Apply NER to avoid spell-correcting known entities
    doc = nlp(joined_text)
    entities = {ent.text for ent in doc.ents}

    corrected_tokens = []
    for token in joined_text.split():
        if token in entities:
            corrected_tokens.append(token)
        else:
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
            corrected_tokens.append(suggestions[0].term if suggestions else token)

    return ' '.join(corrected_tokens)
