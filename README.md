# 📊 GEX Master Pro v3.0

Outil d'analyse Gamma Exposure pour Bitcoin avec signaux de trading exploitables.

## 🚀 Features

- ✅ Calcul GEX en temps réel (Deribit)
- ✅ Analyse de vélocité multi-timeframes (4h + 24h)
- ✅ Détection de pente et accélération
- ✅ Signaux LONG/SHORT avec entry/stop/target
- ✅ Historique persistant (JSON)
- ✅ Export TradingView (Pine Script)
- ✅ Auto-fetch recommandé toutes les heures

## 📦 Installation
```bash
git clone https://github.com/VOTRE_USERNAME/gex-master-pro.git
cd gex-master-pro
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

👉 **https://gex-master-pro.streamlit.app**

## 📖 Usage

1. Ouvrir l'app
2. Configurer horizon (jours) et timeframe (SWING/SCALP)
3. Cliquer sur "CALCULER LE GEX"
4. Lire le signal de trading
5. Copier le code TradingView

## 🧠 Logique de Trading

**SWING (24h+) :**
- Prix < ZG + ZG ↗ → Signal LONG (magnétisme haussier)
- Prix > ZG + ZG ↘ → Signal SHORT (magnétisme baissier)

**SCALP (4h) :**
- Proximity aux Call/Put Walls (< 1.8%)
- Pinning magnétique court terme

## ⚠️ Disclaimer

Outil éducatif. Le trading comporte des risques. Faites vos propres recherches.

## 📞 Support

Issues : https://github.com/VOTRE_USERNAME/gex-master-pro/issues
