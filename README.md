# Regional Flex Optimizer – Quick Start

## 1 · Cloner & créer l’environnement
```bash
git clone https://github.com/mon-org/regional_flex.git
cd regional_flex
conda env create -f environment.yml  # ou pip install -r requirements.txt
conda activate regional_flex          # (ou venv)
```

## 2 · Préparer les données
```
project_root/
├── data/
│   └── processed/         # *.csv ou *.parquet prêt à l’emploi
└── config_master.yaml     # paramètres globaux + capacités régionales
```
Le modèle attend un **pas de temps demi‑horaire** (48 × 365 = 17 520 lignes) et les colonnes :
- `demand` (ou `consumption`, `load`)
- `hydro`, `nuclear`, `thermal_gas`, `thermal_coal`, `biofuel`, … (facultatif si déjà déduits)

## 3 · Lancer une optimisation
### Scénario complet 2022
```bash
python run_regional_flex.py --config config/config_master.yaml --data-dir data/processed --preset full_year --out results/full_year.pkl
```

### Jours types prédéfinis
| Preset               | Période ciblée | Exemple de commande |
|----------------------|----------------|---------------------|
| `winter_weekday`     | 18 janvier 22  | `--preset winter_weekday` |
| `autumn_weekend`     | 9 octobre 22   | … |
| `spring_weekday`     | 12 mai 22      | … |
| `summer_holiday`     | 15 août 22     | … |

### Intervalle sur mesure
```bash
python run_regional_flex.py \
       --start 2022-03-01 --end 2022-03-07 \
       --out results/mars.pkl
```

## 4 · Visualiser les résultats
### Script CLI (PNG)
```bash
python view_flex_results.py --pickle results/full_year.pkl --all-regions --out plots
```
Produit :
- `dispatch_<region>.png` · Aire empilée des techno dispatchables
- `soc_<region>.png` · État de charge des stockages
- `slack_<region>.png` · Slack ±
- `curtail_<region>.png` · Curtailment
- `exchanges_<region>.png` · Flux nets inter‑régions

Option `--all-regions` génère toutes les régions ; `--start/--end` coupe la plage.

### Notebook interactif
1. Ouvrir `regional_flex_dashboard_fixed.ipynb` dans Jupyter Lab.
2. Sélectionner :
   - un pickle dans **Results**
   - la **Region**
   - un preset ou dates custom
3. *Update plots* → affichage dynamique + widgets.

## 5 · Paramétrage avancé
- **Capacités régionales** : section `regional_capacities` du YAML.
- **Coûts variables** : `costs:` (globaux) ou `regional_costs:`.
- **Simplifications** : dans le code, `self.use_simplified_model` + `simplification_options`.

## 6 · FAQ rapide
| Problème | Piste / Fix |
|----------|------------|
| Bande violette énorme | Vérifier filtre des techno et séries nulles (palette + `non_zero` filter). |
| Biofuel illimité | Harmoniser espaces → underscores dans les noms de région + bornage par défaut à 0 MW. |
| Slack massif | Penalty trop bas ou contraintes trop serrées. Ajuster `slack_penalty` ou vérif. ramping/échanges. |

---
© 2025 Théotime Coudray – licence MIT

