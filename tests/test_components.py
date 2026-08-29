"""
test_components.py — Basic component test
"""
import sys
from pathlib import Path

# Add root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.amharic_g2p import normalize_text, text_to_tokens, tokens_to_text, VOCAB_SIZE


def test_g2p():
    text = "ሰላም ዓለም 123"
    norm = normalize_text(text)
    tokens = text_to_tokens(norm)
    decoded = tokens_to_text(tokens)
    print(f"G2P Test: Original: {text} -> Norm: {norm} -> Decoded: {decoded}")
    assert len(tokens) > 0
    assert VOCAB_SIZE > 100


if __name__ == "__main__":
    test_g2p()
    print("✅ All basic component tests passed!")
