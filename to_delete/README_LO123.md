# Letterboxd-KG — LO1–LO3 Pipeline

## Struktur
```
letterboxd-KG/
├─ data/
│  ├─ kg/
│  │  ├─ rerank_by_logical_rules.csv
│  │  └─ outputs/  # generierte Exporte
│  └─ letterboxd_export/  # ratings.csv / diary.csv / watchlist.csv
└─ gnn/
   └─ gnn_rerank_pipeline_LO3_full.ipynb
```

## Setup (Repro)
```bash
mamba env create -f environment.yml
mamba activate letterboxd-kg
jupyter notebook
```

## Run
1. Öffne `gnn/gnn_rerank_pipeline_LO3_full.ipynb`.
2. Stelle sicher, dass die Pfade in der ersten Zelle stimmen.
3. Führe alle Zellen aus:
   - Daten laden
   - Labels & Zeit-Split
   - **Baselines (LO1, LO1+LO2)**
   - **GNN mit edge weights (LO3)**
   - **Ablation & Ensemble mit λ-Sweep**
4. Ergebnisse findest du in `data/kg/outputs/`:
   - `ablation_results.csv`
   - `top10_lo1_kge.csv`, `top10_lo1_lo2_baseline.csv`, `top10_lo3_gnn.csv`
   - `top10_ensemble_lam0.2.csv`, ..., `top10_ensemble_lam0.8.csv`

## Hinweise
- GNN nutzt **GATv2Conv** mit `edge_attr`=normalisierte Kanten-Gewichte aus `cos`, `final`, `comp_*`.
- Seeds sind fixiert (SEED=42). Für volle Reproduzierbarkeit bitte gleiche Pakete/Versionen verwenden.
