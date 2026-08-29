"""
amharic_g2p.py — Amharic Grapheme-to-Phoneme & Text Normalization
==================================================================
Handles the full Ethiopic (Ge'ez / Fidel) Unicode block:
  - Basic Ethiopic:       U+1200–U+137F
  - Ethiopic Supplement:  U+1380–U+139F
  - Ethiopic Extended:    U+2D80–U+2DDF
  - Ethiopic Extended-A:  U+AB01–U+AB2F

The Amharic writing system uses an abugida (syllabary). Each character
(fidel) encodes a consonant + vowel combination. This module maps each
fidel to a (consonant, vowel) pair, enabling phoneme-level processing.
"""

import re
import unicodedata
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# 1. Amharic Character Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Special tokens
PAD_TOKEN   = "<pad>"
UNK_TOKEN   = "<unk>"
BOS_TOKEN   = "<bos>"   # beginning of sequence
EOS_TOKEN   = "<eos>"   # end of sequence
SIL_TOKEN   = "<sil>"   # silence / space

# Amharic vowel orders (each Ethiopic consonant has 7 forms)
VOWEL_ORDERS = ["ə", "u", "i", "a", "e", "ɨ", "o"]  # IPA approximations
VOWEL_NAMES  = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

# ─────────────────────────────────────────────────────────────────────────────
# Core Ethiopic consonant roots and their Unicode base code points
# Each base + 0..6 offset = the 7 vowel forms
# ─────────────────────────────────────────────────────────────────────────────
ETHIOPIC_CONSONANTS: Dict[str, int] = {
    "h":  0x1200,  # ሀ ሁ ሂ ሃ ሄ ህ ሆ
    "l":  0x1208,  # ለ ሉ ሊ ላ ሌ ል ሎ
    "ħ":  0x1210,  # ሐ ሑ ሒ ሓ ሔ ሕ ሖ
    "m":  0x1218,  # መ ሙ ሚ ማ ሜ ም ሞ
    "s̠":  0x1220,  # ሠ ሡ ሢ ሣ ሤ ሥ ሦ
    "r":  0x1228,  # ረ ሩ ሪ ራ ሬ ር ሮ
    "s":  0x1230,  # ሰ ሱ ሲ ሳ ሴ ስ ሶ
    "ʃ":  0x1238,  # ሸ ሹ ሺ ሻ ሼ ሽ ሾ
    "q":  0x1240,  # ቀ ቁ ቂ ቃ ቄ ቅ ቆ
    "b":  0x1260,  # በ ቡ ቢ ባ ቤ ብ ቦ
    "v":  0x1268,  # ቨ ቩ ቪ ቫ ቬ ቭ ቮ
    "t":  0x1270,  # ተ ቱ ቲ ታ ቴ ት ቶ
    "tʃ": 0x1278,  # ቸ ቹ ቺ ቻ ቼ ች ቾ
    "x":  0x1290,  # ነ ኑ ኒ ና ኔ ን ኖ — mapped to n here; 'x' is placeholder
    "n":  0x1290,  # ነ ኑ ኒ ና ኔ ን ኖ
    "ɲ":  0x1298,  # ኘ ኙ ኚ ኛ ኜ ኝ ኞ
    "a":  0x12A0,  # አ ኡ ኢ ኣ ኤ እ ኦ (glottal)
    "k":  0x12A8,  # ከ ኩ ኪ ካ ኬ ክ ኮ
    "x2": 0x12C8,  # ወ ዉ ዊ ዋ ዌ ው ዎ — w
    "w":  0x12C8,
    "ʔ":  0x12D0,  # ዐ ዑ ዒ ዓ ዔ ዕ ዖ (pharyngeal)
    "z":  0x12D8,  # ዘ ዙ ዚ ዛ ዜ ዝ ዞ
    "ʒ":  0x12E0,  # ዠ ዡ ዢ ዣ ዤ ዥ ዦ
    "j":  0x12E8,  # የ ዩ ዪ ያ ዬ ይ ዮ
    "d":  0x12F0,  # ደ ዱ ዲ ዳ ዴ ድ ዶ
    "dʒ": 0x12F8,  # ጀ ጁ ጂ ጃ ጄ ጅ ጆ
    "g":  0x1308,  # ገ ጉ ጊ ጋ ጌ ግ ጎ
    "ŋ":  0x1330,  # ጰ ጱ ጲ ጳ ጴ ጵ ጶ — ejective p
    "p'": 0x1330,
    "ts'":0x1338,  # ጸ ጹ ጺ ጻ ጼ ጽ ጾ — ejective ts
    "ts":0x1338,
    "f":  0x1348,  # ፈ ፉ ፊ ፋ ፌ ፍ ፎ
    "p":  0x1350,  # ፐ ፑ ፒ ፓ ፔ ፕ ፖ
}

# Build char→phoneme lookup: code_point → (consonant_phoneme, vowel_phoneme)
CHAR_TO_PHONEME: Dict[str, Tuple[str, str]] = {}

def _build_phoneme_table():
    for consonant, base in ETHIOPIC_CONSONANTS.items():
        if consonant in ("x", "x2"):
            continue  # skip aliases
        for vowel_idx, vowel in enumerate(VOWEL_ORDERS):
            code_point = base + vowel_idx
            char = chr(code_point)
            CHAR_TO_PHONEME[char] = (consonant, vowel)

_build_phoneme_table()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Full Character Vocabulary (for model tokenization)
# ─────────────────────────────────────────────────────────────────────────────

def build_amharic_vocab() -> Dict[str, int]:
    """
    Build a complete character-level vocabulary covering:
    - All Ethiopic fidel characters (U+1200–U+137F)
    - Ethiopic punctuation
    - Common numerals and space
    - Special tokens

    Returns:
        dict mapping character → integer index
    """
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        BOS_TOKEN: 2,
        EOS_TOKEN: 3,
        SIL_TOKEN: 4,
        " ":        5,
    }
    idx = 6

    # All Ethiopic Unicode characters U+1200–U+137F
    for cp in range(0x1200, 0x1380):
        char = chr(cp)
        if unicodedata.category(char) not in ("Cn", "Co"):  # skip unassigned
            vocab[char] = idx
            idx += 1

    # Ethiopic Supplement U+1380–U+139F
    for cp in range(0x1380, 0x13A0):
        char = chr(cp)
        if unicodedata.category(char) not in ("Cn", "Co"):
            vocab[char] = idx
            idx += 1

    # Ethiopic Extended U+2D80–U+2DDF
    for cp in range(0x2D80, 0x2DE0):
        char = chr(cp)
        if unicodedata.category(char) not in ("Cn", "Co"):
            vocab[char] = idx
            idx += 1

    # Ethiopic punctuation common in Amharic text
    for char in ["፡", "።", "፣", "፤", "፥", "፦", "፧", "፨", "!", "?", ",", "."]:
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    # ASCII digits (appear in mixed text)
    for d in "0123456789":
        vocab[d] = idx
        idx += 1

    return vocab


# Singleton vocab instance
AMHARIC_VOCAB: Dict[str, int] = build_amharic_vocab()
AMHARIC_VOCAB_INV: Dict[int, str] = {v: k for k, v in AMHARIC_VOCAB.items()}
VOCAB_SIZE: int = len(AMHARIC_VOCAB)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

# Amharic digits (Ethiopic numerals)
AMHARIC_NUMERALS = {
    "0": "ዜሮ", "1": "አንድ", "2": "ሁለት", "3": "ሶስት",
    "4": "አራት", "5": "አምስት", "6": "ስድስት", "7": "ሰባት",
    "8": "ስምንት", "9": "ዘጠኝ",
}

# Common abbreviations used in Amharic text
AMHARIC_ABBREVIATIONS = {
    "ዶ/ር":  "ዶክተር",
    "ፕ/ሚ":  "ጠቅላይ ሚኒስትር",
    "ኮ/ል":  "ኮሎኔል",
    "ወ/ሮ":  "ወይዘሮ",
    "ወ/ት":  "ወይዘሪት",
    "አ.አ":  "አዲስ አበባ",
}


def normalize_text(text: str) -> str:
    """
    Normalize Amharic text for TTS/ASR training.

    Steps:
    1. Expand common Amharic abbreviations
    2. Convert ASCII digits to Amharic words
    3. Replace Ethiopic word separator (፡) with space
    4. Remove unsupported characters
    5. Collapse multiple spaces

    Args:
        text: Raw Amharic text string

    Returns:
        Normalized text string
    """
    # Step 1: Expand abbreviations
    for abbr, expansion in AMHARIC_ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)

    # Step 2: Convert digits to Amharic words
    for digit, word in AMHARIC_NUMERALS.items():
        text = text.replace(digit, f" {word} ")

    # Step 3: Normalize Ethiopic punctuation
    text = text.replace("፡", " ")   # Ethiopic word separator → space
    text = text.replace("።", ".")   # Ethiopic full stop → ASCII period

    # Step 4: Strip characters not in vocabulary
    allowed = set(AMHARIC_VOCAB.keys()) | {" ", ".", ",", "!", "?"}
    text = "".join(c for c in text if c in allowed)

    # Step 5: Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def text_to_tokens(text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
    """
    Convert normalized Amharic text to a list of token IDs.

    Args:
        text:    Normalized Amharic text
        add_bos: Prepend BOS token
        add_eos: Append EOS token

    Returns:
        List of integer token IDs
    """
    tokens = []
    if add_bos:
        tokens.append(AMHARIC_VOCAB[BOS_TOKEN])

    for char in text:
        tokens.append(AMHARIC_VOCAB.get(char, AMHARIC_VOCAB[UNK_TOKEN]))

    if add_eos:
        tokens.append(AMHARIC_VOCAB[EOS_TOKEN])

    return tokens


def tokens_to_text(tokens: List[int]) -> str:
    """
    Decode a list of token IDs back to text string.

    Args:
        tokens: List of integer token IDs

    Returns:
        Decoded text (special tokens stripped)
    """
    special = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SIL_TOKEN}
    chars = []
    for t in tokens:
        char = AMHARIC_VOCAB_INV.get(t, UNK_TOKEN)
        if char not in special:
            chars.append(char)
    return "".join(chars)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Phoneme Sequence (for Tacotron2 phoneme mode)
# ─────────────────────────────────────────────────────────────────────────────

def text_to_phonemes(text: str) -> List[str]:
    """
    Convert Amharic text to a list of phoneme strings.
    Each Ethiopic character → (consonant, vowel) tuple.
    Spaces and punctuation are preserved as-is.

    Args:
        text: Normalized Amharic text

    Returns:
        List of phoneme strings, e.g. ["s", "ə", "l", "a", "m"]
    """
    phonemes = []
    for char in text:
        if char == " ":
            phonemes.append(SIL_TOKEN)
        elif char in CHAR_TO_PHONEME:
            consonant, vowel = CHAR_TO_PHONEME[char]
            phonemes.append(consonant)
            phonemes.append(vowel)
        else:
            # Punctuation or unknown → keep as is for prosody
            phonemes.append(char)
    return phonemes


# ─────────────────────────────────────────────────────────────────────────────
# 6. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_amharic(text: str) -> bool:
    """Return True if the string contains at least one Ethiopic character."""
    return any("\u1200" <= c <= "\u137F" or "\u2D80" <= c <= "\u2DDF" for c in text)


def char_count(text: str) -> Dict[str, int]:
    """Return frequency count of each Amharic character in the text."""
    counts: Dict[str, int] = {}
    for ch in text:
        if is_amharic(ch):
            counts[ch] = counts.get(ch, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 7. Quick Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "ሰላም ዓለም",         # Hello World
        "እንደምን አለህ",       # How are you?
        "ኢትዮጵያ",           # Ethiopia
        "ዶ/ር አብርሃም",       # Dr. Abraham (with abbreviation)
        "1 ፡ 2 ፡ 3",       # Digits + Ethiopic separators
    ]

    print(f"Vocab size: {VOCAB_SIZE}")
    print("-" * 50)

    for text in sample_texts:
        normalized = normalize_text(text)
        tokens     = text_to_tokens(normalized)
        decoded    = tokens_to_text(tokens)
        phonemes   = text_to_phonemes(normalized)

        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print(f"Tokens:     {tokens[:10]}{'...' if len(tokens)>10 else ''}")
        print(f"Decoded:    {decoded}")
        print(f"Phonemes:   {phonemes[:10]}{'...' if len(phonemes)>10 else ''}")
        print()
