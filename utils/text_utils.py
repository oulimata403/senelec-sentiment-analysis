"""
Utilitaires pour traitement de texte
"""
import re
import string
from typing import List

def clean_text(text: str) -> str:
    """Nettoyage basique de texte"""
    if not isinstance(text, str):
        return ""
    
    # Minuscules
    text = text.lower()
    
    # Supprimer URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Supprimer mentions
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer hashtags (garder le mot)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Supprimer ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """Tokenisation simple"""
    return text.split()

def remove_short_words(text: str, min_length: int = 3) -> str:
    """Supprime mots trop courts"""
    words = text.split()
    filtered = [w for w in words if len(w) >= min_length]
    return ' '.join(filtered)