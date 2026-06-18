# chinese-braille-mt5

Chinese Braille-to-text translation toolkit with data preprocessing, dataset generation, sequence disambiguation, and mT5-based neural translation.

This repository provides a pipeline for building Chinese Braille translation datasets and training an mT5-based sequence-to-sequence model to convert Chinese Braille text into ordinary Chinese text. It includes data preparation scripts, Braille-specific tokenizer expansion, model fine-tuning, evaluation, inference, and structured Braille decoding utilities.

## Features

* Chinese Braille-to-text translation based on mT5
* Braille-specific tokenizer vocabulary expansion
* Chinese text slicing and Braille dataset generation
* Train / validation / test split generation
* Tone-mark removal for robustness-oriented data augmentation
* Structured Braille cell decoding
* Candidate expansion and sequence disambiguation
* BLEU-based evaluation and simple inference demo

## Project Structure

```text
chinese-braille-mt5/
├── README.md
├── requirements.txt
├── mt5_add_special_tokens.py
├── run_translation.py
├── run_translation_accelerate.sh
├── run_translation_evaluation.sh
├── test_inference_simp.py
├── data_preparation/
│   ├── README.md
│   ├── slicing.py
│   ├── braille_convert.py
│   ├── combine_processed_data.py
│   ├── braille_dataset_generation.py
│   └── helper_allign_braille_data.py
└── decoder/
    ├── structured_braille_decoder.py
    ├── candidate_expansion.py
    └── sequence_disambiguation.py
```

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
torch
accelerate
transformers
datasets
evaluate
tensorboard
sacrebleu
```

## Dataset Preparation

The data preparation scripts are located in the `data_preparation/` directory.

### 1. Prepare raw Chinese text

Place the original Chinese corpus file under the `data/` directory. For example:

```text
data/zho_news_2007-2009_1M-sentences.txt
```

### 2. Slice large text files

Large text files can be split into smaller files before Braille conversion:

```bash
python data_preparation/slicing.py \
  --input_filename ./data/zho_news_2007-2009_1M-sentences.txt \
  --output_filename_base ./sliced_data/processed_input \
  --slice_size 1000
```

### 3. Convert Chinese text into Braille

Use the Braille conversion script to process the sliced text files:

```bash
python data_preparation/braille_convert.py \
  --data_dir ./sliced_data \
  --output_dir ./processed_data \
  --max_workers 10
```

### 4. Combine processed data

After Braille conversion, combine the processed Braille files and the corresponding original Chinese files:

```bash
python data_preparation/combine_processed_data.py \
  --braille_path ./processed_data \
  --total_number 1000 \
  --sliced_CN_datadir ./sliced_data
```

This step generates combined data files such as:

```text
data/0_combined_braille.txt
data/0_combined_original.txt
```

### 5. Generate training / validation / testing data

Generate JSON files for model training:

```bash
python data_preparation/braille_dataset_generation.py \
  --output_dir ./data/braille_dataset \
  --remove_tone_portion 0.9
```

The generated dataset contains:

```text
training_data.json
validation_data.json
testing_data.json
```

The parameter `remove_tone_portion` controls the proportion of Braille tone marks removed from the input text. This can be used to improve robustness under incomplete or noisy Braille inputs.

Examples:

```bash
# Keep all tone marks
python data_preparation/braille_dataset_generation.py \
  --output_dir ./data/braille_dataset_full_tone \
  --remove_tone_portion 0

# Remove 90% of tone marks
python data_preparation/braille_dataset_generation.py \
  --output_dir ./data/braille_dataset_90_removed \
  --remove_tone_portion 0.9

# Remove all tone marks
python data_preparation/braille_dataset_generation.py \
  --output_dir ./data/braille_dataset_no_tone \
  --remove_tone_portion 1
```

## Model Preparation

Before training, add Braille characters to the mT5 tokenizer:

```bash
python mt5_add_special_tokens.py \
  --original_model_dir ./models/mt5-small \
  --output_dir ./models/mt5-braille
```

The expanded tokenizer and model will be saved to:

```text
models/mt5-braille
```

## Training

Fine-tune the mT5 model using Accelerate:

```bash
bash run_translation_accelerate.sh
```

The default training script uses:

```text
Model path: ./models/mt5-braille
Dataset path: ./data/braille_dataset
Output path: ./output-dirs/finetune-mt5
```

You can modify hyperparameters such as batch size, number of epochs, maximum sequence length, and evaluation steps in `run_translation_accelerate.sh`.

## Evaluation

Run evaluation and prediction:

```bash
bash run_translation_evaluation.sh
```

Generated predictions will be saved under the output directory, for example:

```text
output-dirs/evaluation-final/generated_predictions.txt
```

The evaluation script reports BLEU scores using SacreBLEU.

## Inference

A simple inference example is provided in:

```text
test_inference_simp.py
```

Run:

```bash
python test_inference_simp.py
```

This script loads the trained model from:

```text
./models/mt5-braille
```

and generates Chinese text from a sample Braille input.

## Structured Braille Decoder

The `decoder/` directory contains utilities for structured Braille decoding and candidate-level correction.

Main modules:

```text
decoder/structured_braille_decoder.py
decoder/candidate_expansion.py
decoder/sequence_disambiguation.py
```

These modules support:

* Braille dot candidate representation
* Braille cell reconstruction
* Low-confidence candidate expansion
* Sequence-level disambiguation
* Character accuracy, CER, and WER computation

## Data Format

The training data follows a JSON format:

```json
[
  {
    "input_text": "⠼⠁ ⠓⠩⠆ ⠵⠪⠆ ...",
    "output_text": "1 原始中文文本 ..."
  }
]
```

Each sample contains:

* `input_text`: Chinese Braille text
* `output_text`: corresponding Chinese text

## Notes

Large datasets, model checkpoints, temporary processed files, and training outputs are not included in this repository by default.

Recommended files or folders to exclude from Git tracking include:

```text
data/
sliced_data/
processed_data/
models/
output-dirs/
__pycache__/
*.log
```

If an external Braille conversion service is used, please make sure the data usage follows the service terms and privacy requirements.

## Citation

If this repository is useful for your research or project, please consider citing or acknowledging this work.

```bibtex
@misc{chinese_braille_mt5,
  title = {Chinese Braille-to-Text Translation with mT5},
  author = {Daxia123p},
  year = {2026},
  howpublished = {\url{https://github.com/daxia123p/chinese-braille-mt5}}
}
```

## License

This project is released under the MIT License.
