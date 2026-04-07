"""
Module des pages du dashboard SENELEC
"""

from .overview import show_overview
from .thematique import show_thematique
from .comparaison import show_comparaison
from .geographie import show_geographie

__all__ = ['show_overview', 'show_thematique', 'show_comparaison', 'show_geographie']