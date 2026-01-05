"""
Analyseur de Vélocité GEX - Version Production
Focus : Pente, évolution, et détection de tendance robuste
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class VelocityAnalyzer:
    """
    Analyse la vélocité et l'accélération du Zero Gamma
    Version simplifiée pour production
    """
    
    def __init__(self, history: List[Dict]):
        """
        Args:
            history: Liste [{timestamp, zero_gamma, spot, ...}]
        """
        self.history = history
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convertit l'historique en DataFrame temporel"""
        if not self.history or len(self.history) < 2:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def calculate_velocity(self, hours: int = 24) -> Dict:
        """
        Calcule la vélocité sur une fenêtre temporelle
        
        Args:
            hours: 4 (scalp) ou 24 (swing)
        
        Returns:
            dict: Résultat complet de l'analyse
        """
        if len(self.df) < 2:
            return self._null_result(hours)
        
        # Filtrer la fenêtre temporelle
        now = self.df['timestamp'].iloc[-1]
        cutoff = now - timedelta(hours=hours)
        window = self.df[self.df['timestamp'] >= cutoff].copy()
        
        if len(window) < 2:
            return self._null_result(hours)
        
        # === VÉLOCITÉ BRUTE ===
        zg_start = window['zero_gamma'].iloc[0]
        zg_end = window['zero_gamma'].iloc[-1]
        
        velocity_raw = zg_end - zg_start
        velocity_pct = (velocity_raw / zg_start * 100) if zg_start != 0 else 0
        
        # === VÉLOCITÉ LISSÉE (EMA) ===
        if len(window) >= 3:
            span = min(len(window) // 3, 5)  # Span adaptatif
            window['zg_ema'] = window['zero_gamma'].ewm(span=span, adjust=False).mean()
            zg_start_smooth = window['zg_ema'].iloc[0]
            zg_end_smooth = window['zg_ema'].iloc[-1]
            velocity_smooth = (zg_end_smooth - zg_start_smooth) / zg_start_smooth * 100
        else:
            velocity_smooth = velocity_pct
        
        # === ACCÉLÉRATION (Pente de la pente) ===
        acceleration = self._calculate_acceleration(window)
        
        # === TENDANCE & FORCE ===
        trend, strength = self._classify_trend(velocity_smooth, hours)
        
        # === CONFIANCE ===
        confidence = self._calculate_confidence(window, velocity_smooth, acceleration)
        
        return {
            'velocity_pct': round(velocity_pct, 2),
            'velocity_smooth': round(velocity_smooth, 2),
            'velocity_raw': round(velocity_raw, 2),
            'acceleration': round(acceleration, 2),
            'trend': trend,
            'strength': strength,
            'confidence': round(confidence, 1),
            'timeframe_hours': hours,
            'data_points': len(window),
            'trend_emoji': self._get_trend_emoji(trend, strength)
        }
    
    def _calculate_acceleration(self, window: pd.DataFrame) -> float:
        """
        Détecte si la vélocité accélère ou décélère
        Méthode : Compare première moitié vs deuxième moitié
        """
        if len(window) < 4:
            return 0.0
        
        mid = len(window) // 2
        first_half = window.iloc[:mid]
        second_half = window.iloc[mid:]
        
        # Vélocité première moitié
        if len(first_half) >= 2:
            vel_1 = (first_half['zero_gamma'].iloc[-1] - first_half['zero_gamma'].iloc[0])
            vel_1_pct = vel_1 / first_half['zero_gamma'].iloc[0] * 100
        else:
            vel_1_pct = 0
        
        # Vélocité deuxième moitié
        if len(second_half) >= 2:
            vel_2 = (second_half['zero_gamma'].iloc[-1] - second_half['zero_gamma'].iloc[0])
            vel_2_pct = vel_2 / second_half['zero_gamma'].iloc[0] * 100
        else:
            vel_2_pct = 0
        
        # Accélération = différence
        return vel_2_pct - vel_1_pct
    
    def _classify_trend(self, velocity: float, hours: int) -> tuple:
        """
        Classifie la tendance avec seuils adaptatifs
        
        Returns:
            (trend, strength): ('BULLISH'|'BEARISH'|'NEUTRAL', 'STRONG'|'MODERATE'|'WEAK')
        """
        # Seuils selon timeframe
        if hours <= 4:  # Scalp
            strong_threshold = 1.5
            moderate_threshold = 0.8
        else:  # Swing 24h
            strong_threshold = 2.5
            moderate_threshold = 1.5
        
        # Classification
        if velocity > strong_threshold:
            return 'BULLISH', 'STRONG'
        elif velocity > moderate_threshold:
            return 'BULLISH', 'MODERATE'
        elif velocity < -strong_threshold:
            return 'BEARISH', 'STRONG'
        elif velocity < -moderate_threshold:
            return 'BEARISH', 'MODERATE'
        else:
            return 'NEUTRAL', 'WEAK'
    
    def _calculate_confidence(self, window: pd.DataFrame, velocity: float, acceleration: float) -> float:
        """
        Score de confiance 0-100
        
        Facteurs:
        - Consistance de la direction (linéarité)
        - Nombre de points de données
        - Magnitude du signal
        - Alignement accélération/vélocité
        """
        confidence = 0.0
        
        # Facteur 1 : Linéarité (R² simplifié)
        if len(window) >= 3:
            x = np.arange(len(window))
            y = window['zero_gamma'].values
            
            # Calcul coefficient de corrélation
            if np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                r_squared = corr ** 2
                confidence += r_squared * 40  # Max 40 points
        
        # Facteur 2 : Nombre de points
        data_score = min(len(window) / 15, 1.0) * 25  # Max 25 points
        confidence += data_score
        
        # Facteur 3 : Magnitude
        mag_score = min(abs(velocity) / 4.0, 1.0) * 20  # Max 20 points
        confidence += mag_score
        
        # Facteur 4 : Alignement accélération/vélocité
        if (velocity > 0 and acceleration > 0) or (velocity < 0 and acceleration < 0):
            confidence += 15  # Bonus convergence
        elif abs(acceleration) < 0.5:
            confidence += 5   # Neutre = OK
        
        return min(confidence, 100)
    
    def _get_trend_emoji(self, trend: str, strength: str) -> str:
        """Emoji visuel pour la tendance"""
        if trend == 'BULLISH':
            return '🚀' if strength == 'STRONG' else '📈'
        elif trend == 'BEARISH':
            return '💥' if strength == 'STRONG' else '📉'
        else:
            return '➡️'
    
    def _null_result(self, hours: int) -> Dict:
        """Résultat par défaut si pas assez de données"""
        return {
            'velocity_pct': 0.0,
            'velocity_smooth': 0.0,
            'velocity_raw': 0.0,
            'acceleration': 0.0,
            'trend': 'NEUTRAL',
            'strength': 'WEAK',
            'confidence': 0.0,
            'timeframe_hours': hours,
            'data_points': 0,
            'trend_emoji': '⚪'
        }
    
    def get_multi_timeframe_analysis(self) -> Dict:
        """
        Analyse complète sur 2 timeframes (4h + 24h)
        
        Returns:
            dict: {scalp_4h, swing_24h, consensus}
        """
        scalp = self.calculate_velocity(hours=4)
        swing = self.calculate_velocity(hours=24)
        
        # === CONSENSUS ===
        # Le timeframe dominant = celui avec la meilleure confiance
        primary = swing if swing['confidence'] > scalp['confidence'] else scalp
        
        # Convergence = les deux vont dans le même sens
        convergence = (scalp['trend'] == swing['trend'] and 
                      scalp['trend'] != 'NEUTRAL')
        
        # Ajustement confiance
        if convergence:
            consensus_confidence = min((swing['confidence'] + scalp['confidence']) / 2 * 1.2, 100)
        else:
            consensus_confidence = primary['confidence']
        
        # Interprétation
        interpretation = self._interpret_consensus(primary, convergence)
        
        consensus = {
            'trend': primary['trend'],
            'strength': primary['strength'],
            'confidence': round(consensus_confidence, 1),
            'convergence': convergence,
            'primary_timeframe': '24h' if primary == swing else '4h',
            'interpretation': interpretation,
            'trend_emoji': primary['trend_emoji']
        }
        
        return {
            'scalp_4h': scalp,
            'swing_24h': swing,
            'consensus': consensus
        }
    
    def _interpret_consensus(self, primary: Dict, convergence: bool) -> str:
        """Génère l'interprétation textuelle pour les traders"""
        trend = primary['trend']
        strength = primary['strength']
        accel = primary['acceleration']
        
        # Base
        if trend == 'BULLISH':
            base = "📈 Le Zero Gamma monte"
            if strength == 'STRONG':
                action = "→ Signal HAUSSIER FORT : Chercher LONG"
            else:
                action = "→ Signal haussier modéré : LONG avec prudence"
        elif trend == 'BEARISH':
            base = "📉 Le Zero Gamma baisse"
            if strength == 'STRONG':
                action = "→ Signal BAISSIER FORT : Chercher SHORT"
            else:
                action = "→ Signal baissier modéré : SHORT avec prudence"
        else:
            base = "➡️ Le Zero Gamma est stable"
            action = "→ Pas de biais directionnel : Range trading ou attendre"
        
        # Accélération
        if abs(accel) > 2:
            if accel > 0 and trend == 'BULLISH':
                accel_note = " ⚡ Momentum s'accélère !"
            elif accel < 0 and trend == 'BEARISH':
                accel_note = " ⚡ Chute s'accélère !"
            else:
                accel_note = " ⚠️ Momentum s'inverse, prudence"
        else:
            accel_note = ""
        
        # Convergence
        if convergence:
            conv_note = " ✅ Convergence 4h/24h"
        else:
            conv_note = " ⚠️ Divergence timeframes"
        
        return f"{base}. {action}{accel_note}{conv_note}"


# === FONCTION UTILITAIRE POUR STREAMLIT ===

def analyze_velocity(history: List[Dict], timeframe: str = 'SWING') -> Dict:
    """
    Point d'entrée simple
    
    Args:
        history: Historique GEX
        timeframe: 'SWING' (24h) ou 'SCALP' (4h)
    
    Returns:
        Résultat de l'analyse
    """
    analyzer = VelocityAnalyzer(history)
    
    if timeframe == 'SCALP':
        return analyzer.calculate_velocity(hours=4)
    else:
        return analyzer.calculate_velocity(hours=24)


def get_full_velocity_analysis(history: List[Dict]) -> Dict:
    """
    Analyse complète (pour dashboard)
    """
    if len(history) < 2:
        return None
    
    analyzer = VelocityAnalyzer(history)
    return analyzer.get_multi_timeframe_analysis()
