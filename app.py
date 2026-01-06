"""
GEX Master Pro v3.0 - Application Principale
Production-ready avec fetch auto et vélocité avancée
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from functools import lru_cache
import time
import json
from pathlib import Path

# Imports modules custom
from core.velocity_analyzer import VelocityAnalyzer, get_full_velocity_analysis
from trading.signal_generator import generate_trading_signal
from core.data_manager import DataManager, get_data_manager, auto_fetch_check, format_time_ago


# === CONFIGURATION ===
st.set_page_config(
    page_title="GEX Master Pro v3.0",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === STYLING ===
st.markdown("""
<style>
    .stApp {background-color: #0E1117;}
    div.stButton > button {
        width: 100%;
        background-color: #2962FF;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {background-color: #0039CB;}
    [data-testid="stMetricValue"] {font-size: 1.2rem;}
    .success-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #00E676;
        margin: 15px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #e65100 0%, #f57c00 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# === CALCULATEUR BLACK-SCHOLES ===
class GreeksCalculator:
    def __init__(self, risk_free_rate=0.0):
        self.r = risk_free_rate

    def is_last_friday_of_month(self, date):
        """Détection robuste du dernier vendredi"""
        if date.month == 12:
            next_month = date.replace(year=date.year + 1, month=1, day=1)
        else:
            next_month = date.replace(month=date.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        days_to_friday = (last_day.weekday() - 4) % 7
        last_friday = last_day - timedelta(days=days_to_friday)
        
        return date.date() == last_friday.date()

    def calculate(self, contract_data):
        try:
            parts = contract_data['instrument_name'].split('-')
            if len(parts) < 4: 
                return None
            
            date_str = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            expiry = datetime.strptime(date_str, "%d%b%y")
        except:
            return None

        S = contract_data.get('underlying_price', 0)
        K = strike
        sigma = contract_data.get('mark_iv', 0) / 100.0
        
        if S == 0 or sigma == 0 or sigma < 0.01:
            return None

        now = datetime.now()
        T = (expiry - now).total_seconds() / (365 * 24 * 3600)
        
        if T <= 0 or T > 5 or T < 1/365:
            return None

        days_to_expiry = (expiry - now).days
        weekday = expiry.weekday()
        month = expiry.month
        
        is_monthly = (weekday == 4 and self.is_last_friday_of_month(expiry))
        is_quarterly = is_monthly and (month in [3, 6, 9, 12])

        # Black-Scholes
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            if np.isnan(gamma) or np.isinf(gamma):
                return None
                
        except:
            return None
        
        contract_data['greeks'] = {"gamma": round(gamma, 5)}
        contract_data['dte_days'] = days_to_expiry
        contract_data['weekday'] = weekday
        contract_data['is_quarterly'] = is_quarterly
        contract_data['is_monthly'] = is_monthly
        contract_data['expiry_date'] = expiry
        
        return contract_data


# === API AVEC CACHE ===
@lru_cache(maxsize=1)
def get_deribit_data_cached(currency, timestamp):
    """Cache valide 1 minute"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url_spot = f"https://www.deribit.com/api/v2/public/get_index_price?index_name={currency.lower()}_usd"
        spot_res = requests.get(url_spot, headers=headers, timeout=10).json()
        spot = spot_res['result']['index_price']
        
        url_book = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        params = {'currency': currency, 'kind': 'option'}
        book_res = requests.get(url_book, params=params, headers=headers, timeout=10).json()
        data = book_res['result']
        return spot, data
    except Exception as e:
        st.error(f"❌ Erreur API Deribit : {e}")
        return None, None

def get_deribit_data(currency='BTC'):
    """Wrapper avec cache par minute"""
    current_minute = int(time.time() / 60)
    return get_deribit_data_cached(currency, current_minute)


# === ANALYSEUR EXPIRATIONS ===
def analyze_upcoming_expirations(data):
    expirations = {}
    now = datetime.now()
    
    for entry in data:
        try:
            parts = entry['instrument_name'].split('-')
            date_str = parts[1]
            expiry = datetime.strptime(date_str, "%d%b%y")
            days_left = (expiry - now).days
            
            if days_left < 0: 
                continue
            
            date_key = expiry.strftime("%d %b %Y")
            weekday = expiry.weekday()
            day = expiry.day
            month = expiry.month
            is_monthly = (day > 21 and weekday == 4)
            is_quart = is_monthly and (month in [3, 6, 9, 12])
            
            if is_quart and days_left not in expirations:
                expirations[days_left] = {"date": date_key, "type": "👑 Quarterly"}
            elif is_monthly and days_left not in expirations:
                expirations[days_left] = {"date": date_key, "type": "🏆 Monthly"}
        except:
            continue
    
    sorted_days = sorted(expirations.keys())
    return sorted_days, expirations


# === PROCESSING GEX ===
def process_gex(spot, data, dte_limit, only_fridays, use_weighting, w_quart, w_month, w_week):
    calculator = GreeksCalculator()
    strikes = {}
    missed_quarterly_dtes = []
    
    for entry in data:
        contract = calculator.calculate(entry)
        
        if contract is None:
            continue
            
        instr = contract.get('instrument_name', 'UNKNOWN')
        dte = contract.get('dte_days', 9999)
        is_quart = contract.get('is_quarterly', False)
        is_month = contract.get('is_monthly', False)
        weekday = contract.get('weekday', -1)
        oi = contract.get('open_interest', 0)
        greeks = contract.get('greeks')
        
        if dte > dte_limit:
            if is_quart: 
                missed_quarterly_dtes.append(dte)
            continue
        
        if only_fridays and weekday != 4: 
            continue
        if oi == 0: 
            continue
        if not greeks: 
            continue
        
        try:
            parts = instr.split('-')
            if len(parts) < 4: 
                continue
            strike = float(parts[2])
            opt_type = parts[3]
            gamma = greeks.get('gamma', 0) or 0
            
            weight = 1.0
            if use_weighting:
                if is_quart: weight = w_quart
                elif is_month: weight = w_month
                else: weight = w_week
            
            gex_val = ((gamma * oi * (spot ** 2) / 100) / 1_000_000) * weight
            
            if strike not in strikes: 
                strikes[strike] = {'total_gex': 0}
            if opt_type == 'C': 
                strikes[strike]['total_gex'] += gex_val
            else: 
                strikes[strike]['total_gex'] -= gex_val
        except:
            continue
    
    # Warnings
    warnings = []
    if missed_quarterly_dtes:
        next_missed_q = min(missed_quarterly_dtes)
        if next_missed_q < (dte_limit * 2):
            warnings.append(
                f"⚠️ QUARTERLY PROCHE IGNORÉE : Dans {next_missed_q} jours "
                f"(horizon: {dte_limit}j). Augmentez à {next_missed_q + 10}j."
            )
    
    if not strikes:
        return pd.DataFrame(), spot, spot, spot, warnings, 0
    
    df = pd.DataFrame.from_dict(strikes, orient='index')
    df.index.name = 'Strike'
    df = df.sort_index()
    
    # Call/Put Walls
    relevant_df = df[(df.index > spot * 0.7) & (df.index < spot * 1.3)]
    if not relevant_df.empty:
        call_wall = relevant_df['total_gex'].idxmax()
        put_wall = relevant_df['total_gex'].idxmin()
    else:
        call_wall = df['total_gex'].idxmax()
        put_wall = df['total_gex'].idxmin()
    
    # Zero Gamma
    subset = df[(df.index > spot * 0.85) & (df.index < spot * 1.15)]
    
    if subset.empty:
        subset = df[(df.index > spot * 0.5) & (df.index < spot * 2.0)]
    
    neg_gex = subset[subset['total_gex'] < 0]
    pos_gex = subset[subset['total_gex'] > 0]
    zero_gamma = spot
    
    if not neg_gex.empty and not pos_gex.empty:
        idx_neg = neg_gex.index.max()
        val_neg = neg_gex.loc[idx_neg, 'total_gex']
        candidates_pos = pos_gex[pos_gex.index > idx_neg]
        if not candidates_pos.empty:
            idx_pos = candidates_pos.index.min()
            val_pos = candidates_pos.loc[idx_pos, 'total_gex']
            ratio = abs(val_neg) / (abs(val_neg) + val_pos)
            zero_gamma = idx_neg + (idx_pos - idx_neg) * ratio
        else:
            zero_gamma = subset['total_gex'].abs().idxmin()
    else:
        if not subset.empty:
            zero_gamma = subset['total_gex'].abs().idxmin()
    
    # Confidence
    confidence = calculate_confidence(df, zero_gamma, spot)
    
    return df, call_wall, put_wall, zero_gamma, warnings, confidence


def calculate_confidence(df, zero_gamma, spot):
    """Score de confiance 0-100"""
    window = df[(df.index > zero_gamma * 0.98) & (df.index < zero_gamma * 1.02)]
    density_score = min(len(window) * 10, 40)
    
    distance_pct = abs(zero_gamma - spot) / spot
    distance_score = max(0, 30 - distance_pct * 100)
    
    neg_sum = abs(df[df['total_gex'] < 0]['total_gex'].sum())
    pos_sum = df[df['total_gex'] > 0]['total_gex'].sum()
    
    if neg_sum == 0 or pos_sum == 0:
        balance_score = 0
    else:
        ratio = min(neg_sum, pos_sum) / max(neg_sum, pos_sum)
        balance_score = ratio * 30
    
    return min(100, density_score + distance_score + balance_score)


# === GESTION HISTORIQUE ===
def load_history():
    """Charge l'historique via DataManager"""
    dm = get_data_manager()
    return dm.load_history()


def save_gex_snapshot(spot, call_wall, put_wall, zero_gamma, confidence):
    """Sauvegarde snapshot via DataManager"""
    dm = get_data_manager()
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "spot": float(spot),
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "zero_gamma": float(zero_gamma),
        "confidence": float(confidence)
    }
    
    dm.save_snapshot(snapshot)
    return snapshot

# SUITE DE app.py (PARTIE 2/2)

# === INTERFACE STREAMLIT ===

st.title("📊 GEX Master Pro v3.0")
st.caption("Real-time Gamma Exposure Analysis with Velocity Detection")

# === SECTION 1 : STATUS & AUTO-FETCH ===
dm = get_data_manager()
stats = dm.get_statistics()

# Afficher seulement si historique existe
if stats['total_snapshots'] > 0:
    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])

    with col_status1:
        last_update = dm.get_last_update_time()
        if last_update:
            time_ago = format_time_ago(last_update)
            if (datetime.now() - last_update).total_seconds() < 3600:
                st.success(f"📡 {time_ago}")
            else:
                st.warning(f"⚠️ {time_ago}")

    with col_status2:
        # Indicateur auto-fetch
        if auto_fetch_check(dm, interval_minutes=60):
            st.warning("🔄 Fetch recommandé")
        else:
            st.success("✅ À jour")

    with col_status3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # Stats rapides
    st.caption(f"📊 {stats['total_snapshots']} snapshots | "
              f"Durée : {stats['duration_hours']}h | "
              f"Intervalle moy : {stats['avg_update_interval']}min")
else:
    # Premier calcul
    col_first1, col_first2 = st.columns([3, 1])
    
    with col_first1:
        st.info("📡 Aucun historique - Premier calcul requis")
    
    with col_first2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

st.divider()

# === SECTION 2 : EXPIRATIONS ===
# Charger données pour calendrier (cache si déjà fetch)
if 'raw_data' not in st.session_state:
    with st.spinner("🔌 Connexion Deribit..."):
        s, d = get_deribit_data('BTC')
        if s and d:
            st.session_state['spot'] = s
            st.session_state['raw_data'] = d
            st.success("✅ Connecté", icon="✅")
        else:
            st.error("❌ Échec connexion")
            st.stop()

spot = st.session_state['spot']
data = st.session_state['raw_data']

st.markdown("### 📅 Calendrier des Expirations Majeures")

sorted_days, exp_details = analyze_upcoming_expirations(data)

if sorted_days:
    cols = st.columns(min(4, len(sorted_days)))
    for i in range(min(4, len(sorted_days))):
        days = sorted_days[i]
        info = exp_details[days]
        
        color = "🔴" if days < 7 else "🟡" if days < 30 else "🟢"
        
        cols[i].metric(
            label=f"{color} {info['type']}", 
            value=f"{days}j",
            delta=info['date'],
            delta_color="off"
        )

st.divider()

# === SECTION 3 : PARAMÈTRES ===
st.markdown("### ⚙️ Configuration de l'Analyse")

col1, col2, col3 = st.columns(3)

with col1:
    dte_limit = st.slider(
        "📏 Horizon (Jours)", 
        min_value=1, 
        max_value=365, 
        value=65,
        help="Durée maximale des contrats à inclure"
    )

with col2:
    only_fridays = st.checkbox(
        "📆 Vendredis uniquement", 
        value=True,
        help="Filtre pour ne garder que les expirations du vendredi"
    )

with col3:
    use_weighting = st.checkbox(
        "⚖️ Pondération intelligente", 
        value=True,
        help="Donne plus de poids aux quarterly/monthly"
    )

# Poids avancés
with st.expander("🎛️ Réglages avancés"):
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        w_quart = st.number_input("Poids Quarterly", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
    with col_w2:
        w_month = st.number_input("Poids Monthly", value=2.0, min_value=1.0, max_value=10.0, step=0.5)
    with col_w3:
        w_week = st.number_input("Poids Weekly", value=1.0, min_value=0.5, max_value=5.0, step=0.5)

# Sélecteur timeframe
st.markdown("### ⏱️ Timeframe de Trading")

col_tf1, col_tf2 = st.columns(2)

with col_tf1:
    timeframe = st.radio(
        "Horizon temporel",
        options=['SWING', 'SCALP'],
        index=0,
        horizontal=True,
        help="SWING = 24h+ (structurel) | SCALP = 4h (pinning)"
    )

with col_tf2:
    if timeframe == 'SWING':
        st.info("📊 **SWING** : Basé sur vélocité ZG 24h + structure")
    else:
        st.info("⚡ **SCALP** : Basé sur distance aux Walls + vélocité 4h")

st.divider()

# === SECTION 4 : BOUTON CALCUL ===
if st.button("🚀 CALCULER LE GEX", type="primary", use_container_width=True):
    
    with st.spinner("🧮 Calcul GEX en cours..."):
        df, cw, pw, zg, warns, conf = process_gex(
            spot, data, dte_limit, only_fridays, 
            use_weighting, w_quart, w_month, w_week
        )
    
    # Sauvegarde historique
    save_gex_snapshot(spot, cw, pw, zg, conf)
    
    # Warnings
    if warns:
        for w in warns:
            st.warning(w)
    
    if not df.empty:
        st.markdown("---")
        st.markdown("## 📈 Résultats de l'Analyse")
        
        # === ANALYSE DE VÉLOCITÉ ===
        st.markdown("### 🎯 Analyse de Tendance GEX (Pente & Évolution)")
        
        history = load_history()
        
        if len(history) >= 2:
            velocity_analysis = get_full_velocity_analysis(history)
            
            if velocity_analysis:
                mtf = velocity_analysis
                consensus = mtf['consensus']
                scalp = mtf['scalp_4h']
                swing = mtf['swing_24h']
                
                # Badge consensus
                if consensus['trend'] == 'BULLISH':
                    st.success(f"**{consensus['trend_emoji']} TENDANCE : {consensus['trend']} {consensus['strength']}**")
                elif consensus['trend'] == 'BEARISH':
                    st.error(f"**{consensus['trend_emoji']} TENDANCE : {consensus['trend']} {consensus['strength']}**")
                else:
                    st.warning(f"**{consensus['trend_emoji']} TENDANCE : {consensus['trend']}**")
                
                st.markdown(f"💡 **{consensus['interpretation']}**")
                
                # Métriques
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                
                with col_v1:
                    st.metric(
                        "Confiance Consensus",
                        f"{consensus['confidence']:.0f}%",
                        delta="Convergence ✅" if consensus['convergence'] else "Divergence ⚠️"
                    )
                
                with col_v2:
                    st.metric(
                        f"{scalp['trend_emoji']} Court Terme (4h)",
                        f"{scalp['velocity_smooth']:+.2f}%",
                        delta=f"Accel: {scalp['acceleration']:+.1f}%"
                    )
                
                with col_v3:
                    st.metric(
                        f"{swing['trend_emoji']} Moyen Terme (24h)",
                        f"{swing['velocity_smooth']:+.2f}%",
                        delta=f"Accel: {swing['acceleration']:+.1f}%"
                    )
                
                with col_v4:
                    st.metric(
                        "Timeframe dominant",
                        consensus['primary_timeframe'],
                        delta=f"{swing['data_points']} pts données"
                    )
                
                # Détails collapsible
                with st.expander("🔍 Détails Vélocité"):
                    st.markdown("#### ⚡ Scalp (4h)")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Vélocité brute", f"{scalp['velocity_pct']:+.2f}%")
                    with col_s2:
                        st.metric("Vélocité lissée", f"{scalp['velocity_smooth']:+.2f}%")
                    with col_s3:
                        st.metric("Confiance", f"{scalp['confidence']:.0f}%")
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Swing (24h)")
                    col_w1, col_w2, col_w3 = st.columns(3)
                    with col_w1:
                        st.metric("Vélocité brute", f"{swing['velocity_pct']:+.2f}%")
                    with col_w2:
                        st.metric("Vélocité lissée", f"{swing['velocity_smooth']:+.2f}%")
                    with col_w3:
                        st.metric("Confiance", f"{swing['confidence']:.0f}%")
                
                st.divider()
        else:
            st.info("📊 Pas assez d'historique pour analyse de vélocité. Calculez le GEX plusieurs fois pour activer cette fonctionnalité.")
            # Valeurs par défaut
            velocity_24h = 0.0
            velocity_4h = 0.0
        
        # === MÉTRIQUES GEX ===
        st.markdown(f"### Prix Spot BTC : **${spot:,.2f}**")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "🔴 Call Wall",
                f"${cw:,.0f}",
                delta=f"+{((cw - spot) / spot * 100):.1f}%"
            )
        
        with col_m2:
            st.metric(
                "🟢 Put Wall",
                f"${pw:,.0f}",
                delta=f"{((pw - spot) / spot * 100):.1f}%"
            )
        
        with col_m3:
            zg_delta = ((zg - spot) / spot * 100)
            st.metric(
                "⚖️ Zero Gamma",
                f"${zg:,.0f}",
                delta=f"{zg_delta:+.2f}%"
            )
        
        with col_m4:
            conf_color = "🟢" if conf > 70 else "🟡" if conf > 50 else "🔴"
            st.metric(
                f"{conf_color} Confiance GEX",
                f"{conf:.0f}%",
                delta="Haute" if conf > 70 else "Moyenne" if conf > 50 else "Faible"
            )
        
        st.divider()
        
        # === SIGNAL DE TRADING ===
        st.markdown("### 🎯 Signal de Trading")
        
        # Générer signal avec vélocité
        if len(history) >= 2 and velocity_analysis:
            vel_24h = swing['velocity_smooth']
            vel_4h = scalp['velocity_smooth']
        else:
            vel_24h = 0.0
            vel_4h = 0.0
        
        signal = generate_trading_signal(
            spot, zg, cw, pw,
            vel_24h, vel_4h,
            conf, timeframe
        )
        
        # Affichage signal
        col_r1, col_r2 = st.columns([3, 1])
        
        with col_r1:
            if signal['direction'] == 'LONG':
                st.success(f"**{signal['bias']}**")
            elif signal['direction'] == 'SHORT':
                st.error(f"**{signal['bias']}**")
            else:
                st.warning(f"**{signal['bias']}**")
            
            st.markdown(f"""
📍 **Action** : {signal['action']}

🎯 **Entry Zone** : {signal['entry_zone']}  
🏁 **Target** : {signal['target']}  
🛡️ **Stop-Loss** : {signal['stop_loss']}

💡 **Note** : {signal['note']}
            """)
            
            if signal['wall_warning']:
                st.warning(signal['wall_warning'])
        
        with col_r2:
            st.info(f"""
**Niveau de Risque**  
{signal['risk']}

**Confiance Signal**  
{signal['confidence_label']}  
({signal['confidence']:.0f}%)

**Timeframe**  
{signal['timeframe']}

**Régime**  
{signal['regime']}
            """)
        
        # Détection shift
        shift = dm.detect_shift(threshold_pct=2.0)
        if shift:
            st.markdown("---")
            st.markdown("### 🚨 Shift GEX Détecté")
            
            st.warning(f"""
**{shift['severity']} {shift['direction']}**

- Ancien ZG : ${shift['old_value']:,.0f}  
- Nouveau ZG : ${shift['new_value']:,.0f}  
- Variation : ${shift['shift']:,.0f} ({shift['shift_pct']:+.2f}%)  
- Temps : {shift['time_diff']}
            """)
        
        st.divider()
        
        # === GRAPHIQUE GEX ===
        st.markdown("### 📊 Distribution du Gamma Exposure")
        
        df_chart = df[(df.index > spot * 0.7) & (df.index < spot * 1.3)].reset_index()
        
        base = alt.Chart(df_chart).encode(
            x=alt.X('Strike:Q', 
                   axis=alt.Axis(format='$,.0f', title='Strike Price'),
                   scale=alt.Scale(domain=[spot * 0.85, spot * 1.15]))
        )
        
        bars = base.mark_bar(size=3).encode(
            y=alt.Y('total_gex:Q', title='GEX (Millions)'),
            color=alt.condition(
                alt.datum.total_gex > 0,
                alt.value('#00C853'),
                alt.value('#D50000')
            ),
            tooltip=[
                alt.Tooltip('Strike:Q', format='$,.0f', title='Strike'),
                alt.Tooltip('total_gex:Q', format=',.2f', title='GEX')
            ]
        )
        
        spot_line = alt.Chart(pd.DataFrame({'spot': [spot]})).mark_rule(
            color='yellow',
            strokeWidth=2,
            strokeDash=[5, 5]
        ).encode(x='spot:Q')
        
        zg_line = alt.Chart(pd.DataFrame({'zg': [zg]})).mark_rule(
            color='white',
            strokeWidth=2
        ).encode(x='zg:Q')
        
        chart = (bars + spot_line + zg_line).properties(
            height=400
        ).configure_view(
            strokeWidth=0
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        st.divider()
        
        # === HISTORIQUE ZG ===
        if len(history) > 5:
            st.markdown("### 📈 Évolution du Zero Gamma (48h)")
            
            df_hist = pd.DataFrame(history[-48:])
            df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
            
            hist_chart = alt.Chart(df_hist).mark_line(
                point=True,
                color='#2962FF',
                strokeWidth=2
            ).encode(
                x=alt.X('timestamp:T', title='Temps'),
                y=alt.Y('zero_gamma:Q', title='Zero Gamma ($)', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('timestamp:T', format='%d %b %H:%M'),
                    alt.Tooltip('zero_gamma:Q', format='$,.0f'),
                    alt.Tooltip('confidence:Q', format='.0f', title='Conf %')
                ]
            ).properties(height=250).interactive()
            
            current_zg_line = alt.Chart(pd.DataFrame({'zg': [zg]})).mark_rule(
                color='yellow',
                strokeDash=[5, 5]
            ).encode(y='zg:Q')
            
            st.altair_chart(hist_chart + current_zg_line, use_container_width=True)
        
        st.divider()
        
        # === EXPORT TRADINGVIEW ===
        st.markdown("### 📋 Code pour TradingView")
        
        tv_code = f"""// ═══════════════════════════════════════════════════════════
// GEX Master Pro v3.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
// Confiance: {conf:.0f}% | Spot: ${spot:,.2f}
// ═══════════════════════════════════════════════════════════

// --- Niveaux GEX ---
float call_wall = {cw}
float put_wall = {pw}
float zero_gamma = {zg}

// --- Signal ---
// Direction: {signal['direction']}
// Bias: {signal['bias']}
// Entry: {signal['entry_zone']}
// Target: {signal['target']}
// Stop: {signal['stop_loss']}

// --- Vélocité ---
// 24h: {vel_24h:+.2f}%
// 4h: {vel_4h:+.2f}%

// --- Colorisation ---
bgcolor(close > zero_gamma ? color.new(color.red, 90) : color.new(color.green, 90))
plot(zero_gamma, "Zero Gamma", color.white, 2)
plot(call_wall, "Call Wall", color.red, 1, plot.style_circles)
plot(put_wall, "Put Wall", color.green, 1, plot.style_circles)"""
        
        col_code1, col_code2 = st.columns([4, 1])
        
        with col_code1:
            st.code(tv_code, language='pine')
        
        with col_code2:
            if st.button("📋 Copier", use_container_width=True):
                st.info("💡 Sélectionnez et copiez le code manuellement")
        
        # Export CSV
        with st.expander("💾 Exporter les données"):
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Télécharger CSV GEX",
                data=csv,
                file_name=f'gex_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )
            
            hist_json = json.dumps(load_history(), indent=2)
            st.download_button(
                label="📥 Télécharger Historique JSON",
                data=hist_json,
                file_name=f'gex_history_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
                mime='application/json',
            )
    
    else:
        st.error("❌ Aucune donnée à afficher. Vérifiez vos filtres.")

else:
    st.info("👆 Configurez les paramètres et cliquez sur **CALCULER LE GEX**")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>📊 GEX Master Pro v3.0 | Data: Deribit API</p>
    <p>⚠️ Trading comporte des risques. Ceci n'est pas un conseil financier.</p>
</div>
""", unsafe_allow_html=True)
