---
title: Traducteur Multilingue FR-EN-EWE
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Traducteur Multilingue (Français, Anglais, Ewe) + Reconnaissance et Synthèse Vocale

Projet Master 1 IA-BD (EPL S2) — système de traduction automatique et vocale entre le
français, l'anglais et l'ewe (langue locale du Togo).

## Fonctionnalités

- **Traduction texte** : Français ↔ Anglais ↔ Ewe (modèle NLLB-200 de Meta)
- **Reconnaissance vocale (STT)** : transcription audio via un modèle Whisper fine-tuné
  sur l'ewe (`crunch_modele_ewe`)
- **Synthèse vocale (TTS)** : lecture audio des traductions
  - Français / Anglais → gTTS (Google Text-to-Speech)
  - Ewe → modèle Coqui VITS natif, entraîné sur le dataset OpenBible
- **Pipeline audio complet** : audio → transcription → traduction → lecture
- **Interface web** : Streamlit, style épuré inspiré de Google Translate
- **Scripts** : prétraitement de corpus, fine-tuning, évaluation BLEU

## Stack technique

| Composant | Technologie |
|---|---|
| Traduction | NLLB-200-distilled-600M (Hugging Face) |
| STT | Whisper fine-tuné (`crunch_modele_ewe`) |
| TTS fr/en | gTTS |
| TTS ewe | Coqui TTS — `tts_models/ewe/openbible/vits` |
| Interface | Streamlit |
| Hébergement | Hugging Face Spaces (Docker) + Hugging Face Hub (modèle STT) |

## Structure du dépôt

── crunch_modele_ewe/                  # Modèle STT (local uniquement, voir section Modèle STT)

├── src/

│   ├── config.py                       # Configuration centralisée (.env)

│   ├── services/

│   │   ├── translation_service.py      # Traduction via NLLB

│   │   ├── stt_service.py              # Transcription audio via Whisper

│   │   └── tts_service.py              # Synthèse vocale (gTTS + Coqui)

│   └── pipelines/

│       └── multilingual_pipeline.py    # Orchestration texte + audio

├── app/

│   └── streamlit_app.py                # Interface utilisateur

├── scripts/

│   ├── prepare_parallel_data.py        # Nettoyage de corpus

│   ├── train_translation_nllb.py       # Fine-tuning traduction

│   ├── evaluate_translation.py         # Évaluation BLEU

│   ├── run_text_translate.py           # Démo CLI texte

│   └── run_audio_translate.py          # Démo CLI audio

├── Dockerfile                          # Build pour Hugging Face Spaces

├── requirements.txt

└── .env.example

## Modèle STT (`crunch_modele_ewe`)

Le modèle Whisper fine-tuné sur l'ewe est trop volumineux pour être versionné dans ce
dépôt Git. Il est hébergé séparément sur le Hugging Face Hub :

👉 https://huggingface.co/josiasetnm/crunch_modele_ewe

`src/config.py` accepte indifféremment :
- un **chemin local** (ex: `./crunch_modele_ewe`) si le dossier existe sur la machine,
- un **identifiant Hugging Face Hub** (ex: `josiasetnm/crunch_modele_ewe`), téléchargé
  automatiquement au premier lancement.

Le comportement est contrôlé par la variable `STT_MODEL_PATH` (voir Installation).

---

## Installation (en local)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Édite ensuite `.env` selon ta situation :

```dotenv
# Option A — tu as déjà le dossier du modèle en local
STT_MODEL_PATH=./crunch_modele_ewe

# Option B — tu préfères le télécharger depuis Hugging Face Hub
STT_MODEL_PATH=josiasetnm/crunch_modele_ewe

TRANSLATION_MODEL_NAME=facebook/nllb-200-distilled-600M
OUTPUT_DIR=./outputs
DEVICE=cpu
```

## Démo rapide (CLI)

### Traduction texte

```bash
$env:PYTHONPATH="."
python scripts/run_text_translate.py --text "Bonjour tout le monde" --source fr --target en
```

### Audio → STT → Traduction

```bash
$env:PYTHONPATH="."
python scripts/run_audio_translate.py --audio "data/raw/test_ewe.wav" --source ewe --target fr
```

### Interface Streamlit

```bash
$env:PYTHONPATH="."
streamlit run app/streamlit_app.py
```

L'app propose deux onglets : **Texte** (traduction écrite avec bouton 🔊 pour écouter)
et **Voix** (micro ou fichier audio → transcription → traduction → écoute).

---

## Prétraitement du corpus

Le script attend un CSV avec les colonnes `fr_text`, `en_text`, `ewe_text` :

```bash
$env:PYTHONPATH="."
python scripts/prepare_parallel_data.py --input data/raw/corpus_tri.csv --output data/processed/corpus_tri_clean.csv
```

## Fine-tuning de la traduction

Exemple pour la paire `fr → ewe` :

```bash
$env:PYTHONPATH="."
python scripts/train_translation_nllb.py \
  --train-csv data/processed/train.csv \
  --eval-csv data/processed/dev.csv \
  --src-col fr_text \
  --tgt-col ewe_text \
  --src-lang fr \
  --tgt-lang ewe \
  --output-dir outputs/nllb-fr-ewe
```

À reproduire pour les 6 paires : `fr↔en`, `fr↔ewe`, `en↔ewe`.

## Évaluation BLEU

```bash
$env:PYTHONPATH="."
python scripts/evaluate_translation.py --csv data/processed/test_fr_ewe.csv --source-col fr_text --target-col ewe_text --source-lang fr --target-lang ewe
```

---

## Déploiement sur Hugging Face Spaces

Ce dépôt est prêt à être déployé sur un Space en SDK **Docker** :

1. Créer le Space :
```bash
   hf repo create <nom-du-space> --repo-type space --space-sdk docker
```
2. Cloner le Space vide et y copier ce projet (sans `.venv/`, `crunch_modele_ewe/`,
   `.env`, `outputs/`).
3. Pousser :
```bash
   git add .
   git commit -m "Déploiement initial"
   git push
```
4. Dans **Settings → Variables and secrets** du Space, définir :

Le `Dockerfile` à la racine construit l'image et lance Streamlit sur le port 8501.

---

## Limites connues

- **TTS ewe** : voix native disponible via Coqui (`tts_models/ewe/openbible/vits`),
  mais entraînée sur un corpus religieux (Bible) — la prosodie peut différer de l'usage
  courant.
- **Corpus ewe** : volume de données plus faible que pour le français/anglais, ce qui
  limite la qualité de traduction sur certains domaines spécifiques.
- **Premier chargement** : les modèles (NLLB, Whisper, Coqui) se téléchargent et se
  chargent en mémoire au premier appel — ça peut prendre 30 secondes à 2 minutes selon
  la machine.

## Conseils pour la présentation

1. Montrer la démo Streamlit : traduction texte, transcription + traduction audio, et
   lecture audio des résultats (🔊).
2. Présenter un tableau BLEU par paire de langues.
3. Expliquer les limites de données ewe et l'usage du transfer learning (NLLB +
   Whisper fine-tuné + Coqui TTS).
4. Évoquer une roadmap d'amélioration : annotation par des locuteurs natifs, élargissement
   du corpus ewe, augmentation par back-translation.

---

**Équipe** — Master 1 IA-BD, EPL S2 — Proposé par Mme GBEDEVI