from typing import List

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Splits text into chunks by words.
    
    Args:
        text (str): The cleaned text to chunk.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
        
    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return chunks
        
    step_size = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
    return chunks
