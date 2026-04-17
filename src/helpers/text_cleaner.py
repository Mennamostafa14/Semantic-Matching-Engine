# helpers/text_cleaner.py
import re


def clean_text(text: str) -> str:
    """
    Normalize raw extracted text from PDF/TXT loaders.

    Removes:
    - Page number artifacts and header/footer noise
    - URLs and email addresses
    - Control characters and unusual Unicode spaces
    - Runs of dashes/underscores used as visual separators
    - Excessive blank lines and horizontal whitespace
    - Short lines that are pure noise (digits, symbols, whitespace only)
    """
    text = re.sub(r'(?i)(page\s*\d+|\-\s*\d+\s*\-)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'[-_]{3,}', ' ', text)
    text = re.sub(r'[\xa0\u2000-\u200f\u2028\u2029]+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [
        line for line in lines
        if len(line) > 2 and not re.fullmatch(r'[\d\s\W]+', line)
    ]
    return '\n'.join(lines).strip()