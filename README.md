# Dementia Speech Analysis

This project performs comprehensive speech analysis for dementia detection using three key components:

1. **Speech-to-Text (STT) and LLM-based Cognitive Status Classification**
2. **Summarization and Concept Coverage from Transcriptions**
3. **Multimodal (Text + Audio) Dementia Classification**



# Usage:

## Dataset

For dataset access:

- Become a member of DemBank to access the TAUKADIAL dataset

- Download the dataset and place within project

- Create and place etadata CSV with `text`, `label`, and `lang` columns in: `TAUKADIAL-24/test.csv`

---

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```
.
├── llm_classifier.py       # LLM-based classification
├── summarizer.py                    # Summarizer + concept checker
├── text-audio-clf.py                # Multimodal classifier
├── TAUKADIAL-24/
│   ├── TAUKADIAL/test/*.wav       # Dataset audio files
│   ├── transcripts/test/*.txt       # Whisper transcriptions
│   ├── test.csv                     # Labels and metadata
│   └── summaries/                   # Output folder for summaries
├── results/                         # Model outputs
└── requirements.txt
```


---


## 1. STT + LLM-Based Cognitive Status Classification

This module uses transcriptions (from Whisper) as input to Large Language Models (LLMs) like LLaMA, FLAN-T5, Mistral, Falcon, GPT-NeoX, and OPT to classify each sample as:
- `NC` (Normal Cognition) or
- `MCI` (Mild Cognitive Impairment)

### Running Example:
```bash
python llm_classifier.py \
    --llm flan-t5 \
    --csv TAUKADIAL-24/test.csv \
    --lang eng \
    --hf_token <your_huggingface_token>
```

### Output:
Console will display model accuracy, macro F1 score, and unweighted average recall (UAR) for each LLM model.

---

## 2. Voice Note Summarization & Concept Analysis

This module summarizes patient speech and checks whether key concepts (based on the picture stimulus) are preserved. It uses:
- `facebook/bart-large-cnn` for summarization
- spaCy for concept extraction
- Predefined concept sets for each image category based on TAUKADIAL dataset images for picture description test (e.g., cookie theft, cat rescue, Norman Rockwell's 'Coming and Going')

### Running Example:
```bash
python summarizer.py \
    --input_folder TAUKADIAL-24/transcripts/test/ \
    -output_folder TAUKADIAL-24/summaries/
```

### Output:
- Individual summaries saved to `--output_folder`
- `summary_results.csv` with concept coverage score for each patient

Each row contains:
- Patient ID
- Sum of relevant concepts mentioned across 3 image descriptions

---

## 3. Multimodal Text + Audio Dementia Classification

This module trains and evaluates a model that uses combinations of audio and text features:
- **Audio**: Wav2Vec2, eGeMAPS, Librosa
- **Text**: BERT embeddings
- **Fusion Models**: MLP, LSTM, Cross-Attention

### Running Example:
```bash
python text-audio-clf.py \
    --train_csv TAUKADIAL-24/train.csv \
    --test_csv TAUKADIAL-24/test.csv \
    --feature_set wave2vec BERT \
    --model cross_atten \
    --optimizer adamw \
    --scheduler cosine \
    --batch_size 16 \
    --gpu 0
```

### Output:
- Model checkpoints and logs saved to `results/<run_name>/`
- Final evaluation printed to console

---

## Command-Line Arguments

| Argument            | Type        | Description                                                            | Default                         |
|---------------------|-------------|------------------------------------------------------------------------|---------------------------------|
| `--train_csv`       | `str`       | Path to the training CSV file                                          | `TAUKADIAL-24/train.csv`        |
| `--test_csv`        | `str`       | Path to the test CSV file                                              | `TAUKADIAL-24/test.csv`         |
| `--model`           | `str`       | Classifier to use: `mlp`, `lstm`, `cross_atten`                        | `mlp`                           |
| `--use_attention`   | `flag`      | Enable self-attention in MLP classifier (only affects `mlp` model)     | `False`                         |
| `--feature_set`     | `list[str]` | Feature types: `wave2vec`, `wavbert`, `egemaps`, `librosa`, `BERT`     | `["wave2vec", "BERT"]`         |
| `--lang`            | `str`       | Language filter: `eng`, `chin`, or `all`                               | `eng`                           |
| `--epochs`          | `int`       | Number of training epochs                                              | `20`                            |
| `--seed`            | `int`       | Random seed                                                            | `42`                            |
| `--gpu`             | `int`       | GPU device ID                                                          | `0`                             |
| `--batch_size`      | `int`       | Training batch size                                                    | `16`                            |
| `--lr`              | `float`     | Learning rate                                                          | `1e-5`                          |
| `--optimizer`       | `str`       | Optimizer: `adam`, `adamw`, or `sgd`                                   | `adamw`                         |
| `--scheduler`       | `str`       | LR scheduler: `cosine` or `plateau`                                    | `cosine`                        |

