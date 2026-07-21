# Piano implementativo — Occupancy Scorer (Stadio B, `0000.fil`)

**Data:** 2026-07-02 · **Stato:** approvato dopo sessione di grilling · **Esecuzione:** solo server (sul server di calcolo)

## 0. Contesto in tre righe

Tutte le linee di scoring frame-level su `0000.fil` sono chiuse o cappate (AE 0.498, MemAE 0.648, top-k rigettato su val, densità latente ~0.5); il test controllato dimostra che l'unico asse separabile ETI-vs-RFI è l'**occupancy di cadenza** (recon al caso 0.462–0.497, embedding supervisionato 0.631). Questo piano costruisce lo scorer di occupancy **strict** a inference-time — il primo plugin dello Stadio B (vetting) della pipeline a due stadi — e l'harness che lo giudica con barre pre-registrate. Il ban ON/OFF è rilassato per 0000.fil (solo scoring, mai training objective; vedi memoria `arch-constraint-no-on-off`).

## 1. Criterio di successo — PRE-REGISTRATO (non negoziabile a posteriori)

| Readout | Metrica | Barra |
|---|---|---|
| **Primario (operazionale)** | TPR a FP ≤ 10% (soglia = 90° percentile degli score sui negativi reali), SNR = 10, split val, n ≥ 500/strato, negativi ≥ 1000 | **TPR ≥ 80%** |
| Curva di sensibilità | TPR per-SNR; floor = SNR minimo con TPR ≥ 50% | riportare (nessuna barra) |
| **Secondario (controllato)** | AUC ETI (ON-only) vs RFI-control (persistente), stesso fondo quiet, energia caliper-matched — direttamente confrontabile con 0.631 (embedding sup.) e 0.497 (recon) | **AUC ≥ 0.75** |

**Regola di decisione:** entrambe le barre fallite → si riapre la decisione pivot/0000.fil. Una sola passata → analisi dei modi di fallimento prima di concludere. Disciplina invariata: checkpoint pinnato per nome esplicito, `--split val`, seed fisso, npz parametrizzato per braccio/checkpoint, mai `last.ckpt`.

## 2. Design dello scorer

### 2.1 Statistica (decisa in grilling: track congiunto)

Frame di cadenza `(96, 1024)` = 6 osservazioni × 16 bin; `on_indices=(0,2,4)`, `off_indices=(1,3,5)`. Per ogni ipotesi di track τ = (canale di partenza c₀, spostamento totale Δ in canali sull'intero frame):

```
canale(t)   = c₀ + round(Δ · t / 95)                     t = 0..95 (timeline assoluta)
m_i(τ)      = mean su t ∈ obs_i di boxcar(mappa[t, canale(t)], w=3)
S_on(τ)     = min su i ∈ {0,2,4} di m_i(τ)               ← strict AND sui 3 ON
S_off(τ)    = max su i ∈ {1,3,5} di m_i(τ)               ← strict assenza sui 3 OFF
C(τ)        = S_on(τ) − S_off(τ)

score(frame) = max su τ di C(τ);  best_track = argmax
```

Proprietà: continuo (AUC/ROC-ready); RFI persistente → C≈0 (S_on≈S_off); RFI intermittente → C≈0 (min-ON crolla se manca in un solo ON — il difetto del cadence scoring loose, corretto); il fondo per-canale (residuo di bandpass) si cancella al primo ordine perché ON e OFF campionano gli stessi canali. Il boxcar w=3 assorbe l'errore di quantizzazione del track e i profili sub-canale (width 1–10 Hz ≈ 0.4–3.6 canali).

### 2.2 Banco di drift

Parametrizzato in **spostamento totale** Δ (canali su 96 bin), non in Hz/s: `Δ ∈ {−1024, −1022, …, +1024}` (passo 2 → deviazione massima dal track vero ≤ 1 canale; 1025 ipotesi). Copre l'intero range fisico window-limited del generatore (`_max_drift_rate` = 1024·df/(96·dt) ≈ **1.633 Hz/s**; conversione Δ = drift·dt/df·96 = drift × 627.2). Solo track interamente dentro la finestra (stesso vincolo di `_sample_start_channel`): il range valido di c₀ dipende dal segno di Δ; nessun caso "track esce dalla finestra" nell'harness (verificato nel generatore). **Flag per Phase 3:** sulla search reale i segnali a drift > 1.633 Hz/s escono dalla finestra da 1024 canali — servirà windowing più largo o stitching; fuori perimetro qui.

Implementazione vettorizzata: matrice indici `(n_drift, 96)` precomputata + gather batched (torch, CPU/GPU) → `(B, n_drift, n_start)` per-obs means via cumsum sui blocchi da 16; costo ~10⁸ ops/frame, banale su GPU.

### 2.3 Mappe di ingresso (ablation, decisa in grilling)

Stesso scorer, stesso protocollo, due bracci:
- **Braccio A — raw:** frame preprocessato (`preprocess_raw`: bandpass_correct + core_transform), nessun modello.
- **Braccio B — residuo:** `(x − MemAE(x))²`, checkpoint pinnato **`outputs/training/20260630_181920_7911978/checkpoints/epoch=017-val_loss=1.3410.ckpt`** (`configs/model/memae.yaml`).

Decide empiricamente se il front-end AE aggiunge qualcosa allo Stadio B. Se A ≥ B, l'AE resta motivato solo dalle morfologie non-narrowband (da ridiscutere, non da nascondere).

### 2.4 Interfaccia plugin (per il futuro `dispersion.py` di 0001)

`OccupancyScorer` implementa un protocollo minimale `VettingFilter` (`score_frames(maps) -> scores, track_infos`; in Phase 3: `apply(candidates) -> candidates`). La statistica per-obs lungo il track è il pezzo **product-specific** (qui narrowband-shaped — il proxy codificabile, per decisione esplicita); la logica min-ON/max-OFF è il pezzo condiviso 0000/0002. Nessuna logica di cadenza entra nel modello o nel training (vincolo architetturale invariato).

## 3. Harness di valutazione

Nuovo script `scripts/debug/occupancy_scorer_test.py`, riusa i mattoni esistenti (`preprocess_raw`, `NarrowbandDriftingGenerator`, `inject_both` / `sample_cadence_signal_params` + `inject_on_only_cadence`, selezione quiet/RFI per quartili di hot-frac, caliper matching da `encode_separation_test`).

### 3.1 Popolazioni (decise in grilling: si inietta SOLO ETI; negativi = reali as-is)

- **Negativi:** ≥ 2000 frame reali di val **così come sono** (nessun filtro hot-frac: includono quiet, RFI persistente e intermittente con le statistiche naturali). Disgiunti dai siti di iniezione.
- **Positivi:** ETI ON-only (`on_indices=(0,2,4)`, track collineare sulla timeline assoluta — verificato in `inject_on_only_cadence`) iniettato in **500 siti quiet** (quartile basso hot-frac) **+ 500 siti RFI-rich** (quartile alto), `snr_list = [3, 5, 7, 10, 15, 20, 30]`, morfologia congelata per sito. Riportare stratificato per fondo — l'iniezione in fondi RFI-rich fa emergere naturalmente il miss-rate da RFI coincidente negli OFF.
- **Readout controllato secondario:** riuso del percorso a due bracci di `eti_vs_rfi_separation_test.py` (ETI vs persistente, stesso fondo quiet, 150 siti × snr_list) scorato con C(τ) e caliper-matched in energia → AUC confrontabile con 0.631/0.497.

### 3.2 Metriche e output

ROC completa sui negativi reali; TPR@FP∈{1%, 10%} per SNR e per strato di fondo; floor (TPR ≥ 50%); AUC controllata; tutto in npz + tabella stampata. Stratificare anche per `t_profile` (constant vs scintillating) e per |drift|.

### 3.3 Diagnostica dei negativi ad alto score (decisa dopo il riframing "candidato ≠ FP")

Salvare stamp + `best_track` (c₀, Δ, per-obs means) dei **top-20 negativi**: classificazione manuale in (a) RFI pattern-coincidente con gli ON — comportamento *corretto*, misura il carico di validazione visiva atteso in Phase 3 (floor teorico ~1/20 delle intermittenze casuali da 3 blocchi); (b) fallimento della logica min/max — bug/debolezza reale. Il tasso FP "vero" = solo i casi (b); la barra del 10% li conta entrambi, quindi è conservativa.

## 4. File

| File | Azione |
|---|---|
| `src/search/vetting/__init__.py` | nuovo — esporta `OccupancyScorer`, protocollo `VettingFilter` |
| `src/search/vetting/occupancy.py` | nuovo — scorer §2 (numpy/torch, device-agnostic, zero dipendenze dal modello) |
| `tests/test_vetting.py` | nuovo — unit test §5 |
| `scripts/debug/occupancy_scorer_test.py` | nuovo — harness §3 (thin, importa da src e dai debug helper esistenti) |
| `configs/search/default.yaml` | estendere con blocco `vetting:` (on_indices, drift_step, boxcar, fp_budget) |

## 5. Unit test (eseguibili in locale — non toccano dati/training)

Su mappe sintetiche costruite a mano (96×1024, rumore bianco + linea additiva):
1. linea ON-only → score alto e `best_track` ≈ (c₀, Δ) veri (±1 canale, ±1 passo drift);
2. linea persistente (6/6) → C ≈ 0;
3. linea intermittente (in 1–2 ON su 3, o in un OFF) → C ≈ 0 / basso;
4. mappa di solo rumore → score ~ distribuzione nulla (media ≈ 0);
5. track vicino ai bordi → ipotesi fuori finestra escluse, nessun wrap-around silenzioso di `roll`;
6. batched == loop frame-per-frame; CPU == GPU (entro tolleranza float);
7. drift ±: convenzione di segno coerente con setigen (drift > 0 → canali crescenti, verificata in `_sample_start_channel`).

## 6. Sequenza di esecuzione

1. Implementare `occupancy.py` + test (locale, ~1 giorno).
2. Harness (locale per la logica, ~1 giorno); smoke test su 10 siti.
3. Run server: braccio A (raw) e braccio B (residuo MemAE pinnato) sullo stesso set (stesso seed=42).
4. Analisi: tabella barre, stratificazioni, diagnostica top-20 negativi.
5. Aggiornare memoria + handoff con esito e decisione (pass → integrazione Phase 3; doppio fail → riapertura pivot).

## 7. Rischi e assi di analisi noti (da guardare, non da nascondere)

- **Scintillazione (40% delle iniezioni, AR(1) depth 0.2–0.6):** un fade profondo in un singolo ON penalizza il min-ON — è fisica, non un bug. Se domina i miss: valutare *a posteriori* (documentandola come variante, non come barra rinegoziata) una soglia 2-su-3 ON, che però alza il floor di coincidenza da ~1/20 a ~1/4 — trade-off da portare al mentor, non da decidere in silenzio.
- **Boxcar vs profili larghi (wings lorentzian/voigt fino a 40 Hz ≈ 14 canali):** w=3 può sotto-integrare; stratificare per f_profile; w è un parametro, non riprogettazione.
- **Normalizzazione per-frame di `core_transform`:** l'iniezione ad alto SNR gonfia leggermente la MAD del frame (auto-soppressione ~uniforme) — effetto identico per tutti gli scorer finora, nessuna azione.
- **Floor di coincidenza:** un negativo che scora alto NON è necessariamente un errore (vedi §3.3) — è la definizione operativa di candidato; a valle lo gestiscono dedup cross-sorgente e ispezione visiva (Phase 3, già in spec).

## 8. Fuori perimetro (esplicito, deciso in grilling)

Retrain β-NLL / whitening del residuo; AE bottleneck sweep (richiesta mentor — piano separato); test zero-cost sull'addressing MemAE; integrazione in `inference.py`/`candidates.py` (dopo il pass); 0002 (stesso scorer, statistica per-obs diversa) e 0001 (`dispersion.py`); qualunque uso della cadenza come training objective.
