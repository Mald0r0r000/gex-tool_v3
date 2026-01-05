"""
Générateur de Signaux de Trading GEX
Logique validée : Prix > ZG + ZG ↗ = Signal ACHAT
"""

from typing import Dict


def generate_trading_signal(spot: float, zero_gamma: float, call_wall: float, put_wall: float,
                            velocity_24h: float, velocity_4h: float, 
                            confidence_gex: float, timeframe: str = 'SWING') -> Dict:
    """
    Génère un signal de trading exploitable
    
    Args:
        spot: Prix BTC actuel
        zero_gamma: Niveau Zero Gamma
        call_wall: Call Wall dominant
        put_wall: Put Wall dominant
        velocity_24h: Vélocité ZG sur 24h (%)
        velocity_4h: Vélocité ZG sur 4h (%)
        confidence_gex: Confiance du calcul GEX (0-100)
        timeframe: 'SWING' ou 'SCALP'
    
    Returns:
        dict: Signal complet avec action, entry, stop, target
    """
    
    # === CLASSIFICATION RÉGIME ===
    distance_to_zg = spot - zero_gamma
    distance_pct = (distance_to_zg / zero_gamma) * 100
    
    # Position relative
    if abs(distance_pct) < 0.5:
        position = 'AT_ZG'
    elif distance_pct > 0:
        position = 'ABOVE_ZG'
    else:
        position = 'BELOW_ZG'
    
    # Régime GEX (stabilisation vs amplification)
    regime = 'STABLE' if position == 'ABOVE_ZG' else 'UNSTABLE'
    
    # Distances aux Walls
    dist_to_cw_pct = abs(call_wall - spot) / spot * 100
    dist_to_pw_pct = abs(put_wall - spot) / spot * 100
    
    # === GÉNÉRATION DU SIGNAL ===
    if timeframe == 'SWING':
        signal = _generate_swing_signal(
            position, regime, distance_pct,
            velocity_24h, spot, zero_gamma, call_wall, put_wall,
            dist_to_cw_pct, dist_to_pw_pct, confidence_gex
        )
    else:  # SCALP
        signal = _generate_scalp_signal(
            position, velocity_4h, spot, zero_gamma, call_wall, put_wall,
            dist_to_cw_pct, dist_to_pw_pct, confidence_gex
        )
    
    return signal


def _generate_swing_signal(position: str, regime: str, distance_pct: float,
                           velocity: float, spot: float, zg: float, 
                           cw: float, pw: float,
                           dist_cw: float, dist_pw: float, conf: float) -> Dict:
    """Signal SWING (focus structurel)"""
    
    # === STRUCTUREL HAUSSIER (ZG monte) ===
    if velocity > 2.5:
        
        if position == 'BELOW_ZG':
            # Setup optimal : Prix sous ZG haussier
            direction = "LONG"
            bias = "🟢 HAUSSIER FORT"
            action = "✅ SETUP PREMIUM : Prix sous ZG en hausse → Magnétisme haussier"
            entry = f"${spot * 0.998:,.0f} - ${zg * 0.995:,.0f}"
            target = f"${zg:,.0f} (ZG) puis ${cw:,.0f} (Call Wall)"
            stop = f"${zg * 0.975:,.0f}"
            risk = "FAIBLE" if conf > 70 else "MOYEN"
            note = f"Prix {distance_pct:.1f}% sous ZG. Rebond probable vers équilibre."
            conf_adj = conf * 1.25
            
        elif position == 'AT_ZG':
            # Prix au pivot
            direction = "LONG"
            bias = "🟢 HAUSSIER - PIVOT"
            action = "📍 Prix AU Zero Gamma haussier → Entrée agressive possible"
            entry = f"${spot:,.0f} (maintenant)"
            target = f"${cw:,.0f} (Call Wall)"
            stop = f"${zg * 0.98:,.0f}"
            risk = "MOYEN"
            note = "Point d'inflexion. Ratio R:R favorable."
            conf_adj = conf * 1.15
            
        else:  # ABOVE_ZG
            if abs(distance_pct) < 2:
                # Consolidation haussière
                direction = "LONG"
                bias = "🟡 HAUSSIER MODÉRÉ"
                action = "⚠️ Prix déjà sur ZG → Attendre pullback ou breakout Call Wall"
                entry = f"Pullback vers ${zg * 0.995:,.0f}"
                target = f"${cw:,.0f}"
                stop = f"${zg * 0.985:,.0f}"
                risk = "MOYEN"
                note = "En extension. Patient = attendre correction."
                conf_adj = conf * 0.95
            else:
                # Sur-extension
                direction = "NEUTRAL"
                bias = "🟠 SUR-EXTENSION"
                action = "🚫 Prix trop loin du ZG → Attendre retour vers équilibre"
                entry = "Attendre pullback"
                target = f"${zg:,.0f} (retour ZG)"
                stop = "N/A"
                risk = "ÉLEVÉ"
                note = "Sur-extension technique. Correction probable."
                conf_adj = conf * 0.7
    
    # === STRUCTUREL BAISSIER (ZG baisse) ===
    elif velocity < -2.5:
        
        if position == 'ABOVE_ZG':
            # Setup optimal : Prix sur ZG baissier
            direction = "SHORT"
            bias = "🔴 BAISSIER FORT"
            action = "✅ SETUP PREMIUM : Prix sur ZG en baisse → Magnétisme baissier"
            entry = f"${zg * 1.005:,.0f} - ${spot * 1.002:,.0f}"
            target = f"${zg:,.0f} (ZG) puis ${pw:,.0f} (Put Wall)"
            stop = f"${zg * 1.025:,.0f}"
            risk = "FAIBLE" if conf > 70 else "MOYEN"
            note = f"Prix {abs(distance_pct):.1f}% sur ZG. Correction probable vers équilibre."
            conf_adj = conf * 1.25
            
        elif position == 'AT_ZG':
            # Prix au pivot
            direction = "SHORT"
            bias = "🔴 BAISSIER - PIVOT"
            action = "📍 Prix AU Zero Gamma baissier → Entrée agressive SHORT"
            entry = f"${spot:,.0f} (maintenant)"
            target = f"${pw:,.0f} (Put Wall)"
            stop = f"${zg * 1.02:,.0f}"
            risk = "MOYEN"
            note = "Point d'inflexion baissier. Ratio R:R favorable."
            conf_adj = conf * 1.15
            
        else:  # BELOW_ZG
            if abs(distance_pct) < 2:
                # Consolidation baissière
                direction = "SHORT"
                bias = "🟡 BAISSIER MODÉRÉ"
                action = "⚠️ Prix déjà sous ZG → Attendre rebond ou breakout Put Wall"
                entry = f"Rebond vers ${zg * 1.005:,.0f}"
                target = f"${pw:,.0f}"
                stop = f"${zg * 1.015:,.0f}"
                risk = "MOYEN"
                note = "En extension baissière. Patient = attendre rebond."
                conf_adj = conf * 0.95
            else:
                # Survente
                direction = "NEUTRAL"
                bias = "🟠 SURVENTE"
                action = "🚫 Prix trop loin du ZG → Attendre rebond technique"
                entry = "Attendre rebond"
                target = f"${zg:,.0f} (retour ZG)"
                stop = "N/A"
                risk = "ÉLEVÉ"
                note = "Survente technique. Rebond probable."
                conf_adj = conf * 0.7
    
    # === STRUCTUREL NEUTRE (ZG stable) ===
    else:
        
        if regime == 'STABLE' and abs(distance_pct) > 1.5:
            # Mean reversion dans zone stable
            direction = "SHORT" if position == 'ABOVE_ZG' else "LONG"
            bias = "🟠 NEUTRE - RANGE"
            action = f"🔄 ZG stable + Prix {'sur' if position == 'ABOVE_ZG' else 'sous'} ZG → Mean reversion"
            entry = f"${spot:,.0f}"
            target = f"${zg:,.0f} (retour ZG)"
            stop = f"${spot * 1.015 if direction == 'LONG' else spot * 0.985:,.0f}"
            risk = "MOYEN"
            note = "Range trading : retour vers équilibre probable."
            conf_adj = conf * 0.85
            
        else:
            # Pas de setup clair
            direction = "NEUTRAL"
            bias = "⚪ NEUTRE STRICT"
            action = "⏸️ Pas de biais structurel → Attendre confirmation directionnelle"
            entry = "Attendre breakout"
            target = "Dépendra de la direction"
            stop = "N/A"
            risk = "ÉLEVÉ"
            note = "Marché en équilibre. Attendre catalyst ou volatilité."
            conf_adj = conf * 0.6
    
    # === WARNING CALL/PUT WALLS ===
    wall_warning = ""
    if dist_cw < 2 and direction == "LONG":
        wall_warning = f"⚠️ Call Wall très proche (${cw:,.0f}, +{dist_cw:.1f}%). Résistance majeure !"
    elif dist_pw < 2 and direction == "SHORT":
        wall_warning = f"⚠️ Put Wall très proche (${pw:,.0f}, -{dist_pw:.1f}%). Support majeur !"
    
    return {
        'direction': direction,
        'bias': bias,
        'action': action,
        'entry_zone': entry,
        'target': target,
        'stop_loss': stop,
        'risk': risk,
        'confidence': min(100, conf_adj),
        'confidence_label': _get_conf_label(conf_adj),
        'note': note,
        'wall_warning': wall_warning,
        'timeframe': 'SWING (24h+)',
        'regime': regime,
        'position': position
    }


def _generate_scalp_signal(position: str, velocity: float, 
                           spot: float, zg: float, cw: float, pw: float,
                           dist_cw: float, dist_pw: float, conf: float) -> Dict:
    """Signal SCALP (focus pinning)"""
    
    # === PINNING VERS CALL WALL ===
    if dist_cw < 1.8 and dist_cw < dist_pw:
        direction = "LONG"
        bias = "🟢 SCALP HAUSSIER"
        action = f"⚡ Call Wall proche ({dist_cw:.1f}%) → Magnétisme haussier court terme"
        entry = f"${spot:,.0f}"
        target = f"${cw:,.0f} (Wall)"
        stop = f"${spot * 0.993:,.0f}"
        risk = "FAIBLE" if dist_cw < 1 else "MOYEN"
        note = f"Scalp vers Call Wall. TP attendu : +{dist_cw:.1f}%"
        conf_adj = conf * 1.1
    
    # === PINNING VERS PUT WALL ===
    elif dist_pw < 1.8 and dist_pw < dist_cw:
        direction = "SHORT"
        bias = "🔴 SCALP BAISSIER"
        action = f"⚡ Put Wall proche ({dist_pw:.1f}%) → Magnétisme baissier court terme"
        entry = f"${spot:,.0f}"
        target = f"${pw:,.0f} (Wall)"
        stop = f"${spot * 1.007:,.0f}"
        risk = "FAIBLE" if dist_pw < 1 else "MOYEN"
        note = f"Scalp vers Put Wall. TP attendu : -{dist_pw:.1f}%"
        conf_adj = conf * 1.1
    
    # === PINNING VERS ZERO GAMMA ===
    elif abs((spot - zg) / zg * 100) < 1.5:
        if position == 'BELOW_ZG':
            direction = "LONG"
            action = "⚡ Proche du ZG → Scalp LONG vers équilibre"
        else:
            direction = "SHORT"
            action = "⚡ Proche du ZG → Scalp SHORT vers équilibre"
        
        bias = "🟡 SCALP NEUTRE"
        entry = f"${spot:,.0f}"
        target = f"${zg:,.0f} (ZG)"
        stop = f"${spot * 0.993 if direction == 'LONG' else spot * 1.007:,.0f}"
        risk = "MOYEN"
        note = "Scalp vers équilibre Zero Gamma"
        conf_adj = conf * 0.9
    
    # === PAS DE SETUP SCALP ===
    else:
        direction = "NEUTRAL"
        bias = "⚪ PAS DE SETUP SCALP"
        action = "⏸️ Walls trop loin → Pas de magnétisme exploitable"
        entry = "N/A"
        target = "N/A"
        stop = "N/A"
        risk = "ÉLEVÉ"
        note = f"Call Wall : +{dist_cw:.1f}%, Put Wall : -{dist_pw:.1f}%. Attendre."
        conf_adj = conf * 0.6
    
    # === VELOCITY BOOST ===
    wall_warning = ""
    if abs(velocity) > 1.5:
        if velocity > 0 and direction == "LONG":
            conf_adj = min(100, conf_adj * 1.1)
            wall_warning = "✅ Vélocité 4h haussière : renforce le signal"
        elif velocity < 0 and direction == "SHORT":
            conf_adj = min(100, conf_adj * 1.1)
            wall_warning = "✅ Vélocité 4h baissière : renforce le signal"
        elif direction != "NEUTRAL":
            conf_adj *= 0.9
            wall_warning = "⚠️ Vélocité 4h diverge du pinning : prudence"
    
    return {
        'direction': direction,
        'bias': bias,
        'action': action,
        'entry_zone': entry,
        'target': target,
        'stop_loss': stop,
        'risk': risk,
        'confidence': min(100, conf_adj),
        'confidence_label': _get_conf_label(conf_adj),
        'note': note,
        'wall_warning': wall_warning,
        'timeframe': 'SCALP (4-12h)',
        'regime': 'PINNING',
        'position': position
    }


def _get_conf_label(confidence: float) -> str:
    """Label textuel de confiance"""
    if confidence > 75:
        return "✅ TRÈS HAUTE"
    elif confidence > 60:
        return "✅ HAUTE"
    elif confidence > 45:
        return "⚠️ MOYENNE"
    else:
        return "❌ FAIBLE"
