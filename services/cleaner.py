import re

def clean_text(raw_text: str) -> str:
    """Cleans up the raw extracted text."""
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', raw_text)
    
    # Optional: Remove non-ASCII characters if they cause noise
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()
