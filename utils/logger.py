"""
Configuration du système de logging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configure un logger avec sortie console et fichier
    
    Args:
        name: Nom du logger
        log_file: Chemin fichier log (optionnel)
        level: Niveau de logging
    
    Returns:
        Logger configuré
    """
    # Créer logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    else:
        
        date_str = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            LOGS_DIR / f'{name}_{date_str}.log',
            encoding='utf-8'
        )
    
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger