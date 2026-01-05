"""
Data Manager - Gestion historique et fetch automatique
Persistence JSON pour Streamlit Cloud
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import streamlit as st


class DataManager:
    """
    Gère l'historique GEX avec auto-fetch et persistence JSON
    """
    
    def __init__(self, history_file: str = "data/gex_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Crée le fichier s'il n'existe pas"""
        if not self.history_file.exists():
            self.history_file.write_text(json.dumps([]))
    
    def load_history(self) -> List[Dict]:
        """Charge l'historique depuis le JSON"""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            return history
        except:
            return []
    
    def save_snapshot(self, snapshot: Dict) -> bool:
        """
        Sauvegarde un nouveau snapshot
        
        Args:
            snapshot: {timestamp, zero_gamma, spot, call_wall, put_wall, confidence}
        
        Returns:
            bool: True si sauvegardé, False si dupliqué
        """
        history = self.load_history()
        
        # Vérifier si déjà existant (évite doublons)
        if history:
            last = history[-1]
            # Si même snapshot dans les 5 dernières minutes, skip
            last_time = datetime.fromisoformat(last['timestamp'])
            current_time = datetime.fromisoformat(snapshot['timestamp'])
            
            if (current_time - last_time).total_seconds() < 300:  # 5 min
                return False
        
        # Ajouter le snapshot
        history.append(snapshot)
        
        # Garder max 200 snapshots (économie espace)
        history = history[-200:]
        
        # Sauvegarder
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    
    def get_last_update_time(self) -> Optional[datetime]:
        """Retourne le timestamp du dernier snapshot"""
        history = self.load_history()
        
        if not history:
            return None
        
        return datetime.fromisoformat(history[-1]['timestamp'])
    
    def should_auto_fetch(self, interval_minutes: int = 60) -> bool:
        """
        Détermine si un nouveau fetch est nécessaire
        
        Args:
            interval_minutes: Intervalle entre les fetch (défaut 60 min)
        
        Returns:
            bool: True si doit fetch
        """
        last_update = self.get_last_update_time()
        
        if last_update is None:
            return True  # Jamais fetch
        
        now = datetime.now()
        elapsed = (now - last_update).total_seconds() / 60  # En minutes
        
        return elapsed >= interval_minutes
    
    def detect_shift(self, threshold_pct: float = 2.0) -> Optional[Dict]:
        """
        Détecte les shifts significatifs du Zero Gamma
        
        Args:
            threshold_pct: Seuil de variation pour alerte (%)
        
        Returns:
            dict ou None: Détails du shift si détecté
        """
        history = self.load_history()
        
        if len(history) < 2:
            return None
        
        last = history[-1]
        previous = history[-2]
        
        # Calcul du shift
        zg_shift = last['zero_gamma'] - previous['zero_gamma']
        zg_shift_pct = (zg_shift / previous['zero_gamma']) * 100
        
        if abs(zg_shift_pct) > threshold_pct:
            time_diff = (datetime.fromisoformat(last['timestamp']) - 
                        datetime.fromisoformat(previous['timestamp']))
            
            return {
                'type': 'Zero Gamma',
                'old_value': previous['zero_gamma'],
                'new_value': last['zero_gamma'],
                'shift': zg_shift,
                'shift_pct': zg_shift_pct,
                'time_diff': str(time_diff).split('.')[0],  # Format HH:MM:SS
                'severity': '🚨 CRITIQUE' if abs(zg_shift_pct) > 5 else '⚠️ IMPORTANT',
                'direction': '⬆️ HAUSSE' if zg_shift > 0 else '⬇️ BAISSE'
            }
        
        return None
    
    def get_statistics(self) -> Dict:
        """Statistiques globales sur l'historique"""
        history = self.load_history()
        
        if not history:
            return {
                'total_snapshots': 0,
                'first_snapshot': None,
                'last_snapshot': None,
                'avg_update_interval': None
            }
        
        first = datetime.fromisoformat(history[0]['timestamp'])
        last = datetime.fromisoformat(history[-1]['timestamp'])
        
        # Intervalle moyen entre updates
        if len(history) > 1:
            total_duration = (last - first).total_seconds()
            avg_interval = total_duration / (len(history) - 1) / 60  # En minutes
        else:
            avg_interval = None
        
        return {
            'total_snapshots': len(history),
            'first_snapshot': first,
            'last_snapshot': last,
            'avg_update_interval': round(avg_interval, 1) if avg_interval else None,
            'duration_hours': round((last - first).total_seconds() / 3600, 1)
        }
    
    def cleanup_old_data(self, keep_days: int = 7):
        """
        Nettoie les données plus anciennes que X jours
        
        Args:
            keep_days: Nombre de jours à conserver
        """
        history = self.load_history()
        
        if not history:
            return
        
        cutoff = datetime.now() - timedelta(days=keep_days)
        
        # Filtrer
        history = [
            s for s in history 
            if datetime.fromisoformat(s['timestamp']) > cutoff
        ]
        
        # Sauvegarder
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


# === FONCTIONS UTILITAIRES POUR STREAMLIT ===

@st.cache_resource
def get_data_manager():
    """Singleton du DataManager (cache Streamlit)"""
    return DataManager()


def auto_fetch_check(data_manager: DataManager, interval_minutes: int = 60) -> bool:
    """
    Vérifie si un auto-fetch est nécessaire
    Utilisé dans l'UI pour afficher une notification
    
    Returns:
        bool: True si fetch recommandé
    """
    return data_manager.should_auto_fetch(interval_minutes)


def format_time_ago(dt: datetime) -> str:
    """
    Formate un datetime en "il y a X min/heures"
    
    Args:
        dt: datetime à formatter
    
    Returns:
        str: "Il y a 5 min" ou "Il y a 2h"
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "À l'instant"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"Il y a {minutes} min"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"Il y a {hours}h {minutes}min" if minutes > 0 else f"Il y a {hours}h"
    else:
        days = int(seconds / 86400)
        return f"Il y a {days} jour{'s' if days > 1 else ''}"
