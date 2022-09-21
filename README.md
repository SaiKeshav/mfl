# MFL

The repository contains code for training the ReFCoG model on the task of Multilingual Fact Linking

It involves training a dual encoder and a generation model

## Install requirements

```
cd code
pip install -r requirements.txt
conda install -c pytorch faiss-gpu
cd ..

cd GENRE
pip install -r requirements.txt
cd ..
```

Install SentencePiece library from https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source


Install fairseq version used by mGENRE:
```
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq
pip install --editable ./
```


## Data

Download the IndicLink data for testing and WebRED data for training using the following command -
```
cd code/
bash download_data.sh
```

## Training Dual Encoder model

Trains the Dual Encoder model using sentence_transformer library and generates the data for training Seq2Seq models.

```
DE_NAME=de_all
cd code/
python dual_encoder.py --de_mode train.test --model_save_path ../models/$DE_NAME --input_languages indiclink --fact_set all --accumulate_scores sum --topk 10 --build_index --input_languages en
cd ..
```

## Training Seq2Seq model
1. Download pre-trained mGENRE model

```
cd models
wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
tar -xvf fairseq_multilingual_entity_disambiguation.tar.gz
cd ..
```

2. Preprocess data
```
cd GENRE
bash scripts_mgenre/preprocess_fairseq.sh ../models/$DE_NAME/s2s_k10_en/fact ../models/fairseq_multilingual_entity_disambiguation
```

3. Train model
```
NAME=../models/$DE_NAME/s2s_k10_en/fact bash scripts_genre/train.sh
```


4. Run inference
```
cd GENRE
python scripts_mgenre/evaluate_kilt_dataset.py --model_path ../models/$DE_NAME/s2s_k10_en/fact --input_path ../models/$DE_NAME/s2s_k10_en/fact/test --output_path ../models/$DE_NAME/s2s_k10_en/fact/const-fact-predictions.jsonl --checkpoint_file checkpoint_best.pt --trie ../data/fact_trie.pkl --order title_lang --verbose --beams 5
```

5. Compute Scores
```
cd ../code/
python scores.py --model_dir $DATASET  --gold_macro
python scores.py --model_dir $DATASET --recall --topk 5
```

