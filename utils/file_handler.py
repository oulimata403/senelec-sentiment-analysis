"""
Utilitaires pour gestion de fichiers
"""
import pandas as pd
import json
from pathlib import Path
import pickle

def save_csv(df, filepath, **kwargs):
    """Sauvegarde DataFrame en CSV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8', **kwargs)
    print(f"✅ Sauvegardé : {filepath}")

def load_csv(filepath, **kwargs):
    """Charge CSV en DataFrame"""
    return pd.read_csv(filepath, **kwargs)

def save_json(data, filepath):
    """Sauvegarde dictionnaire en JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Sauvegardé : {filepath}")

def load_json(filepath):
    """Charge JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(obj, filepath):
    """Sauvegarde objet en pickle"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Sauvegardé : {filepath}")

def load_pickle(filepath):
    """Charge objet pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)