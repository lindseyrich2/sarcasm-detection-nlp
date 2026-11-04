# Sarcasm Detection with NLP

Detecting sarcasm in Reddit comments by comparing traditional machine learning, recurrent and convolutional networks, and transformer models. A comment's sarcasm often depends on what it replies to, so the models also use the parent comment.

This was a team project for DS 440. The original repository is [brianellis1997/Sarcasm_Detection](https://github.com/brianellis1997/Sarcasm_Detection).

## Objective

Classify a Reddit comment as sarcastic or not, and find out how much each of these adds to performance: conversational context, non-text features, learned embeddings, and transformer architectures.

## Dataset

The notebooks use a balanced Reddit sarcasm dataset, `train-balanced-sarcasm.csv`, with 1,010,773 labelled comments. It isn't included here.

- **Fields:** `label`, `comment`, `parent_comment`, `author`, `subreddit`, `score`, `ups`, `downs`, `date` and `created_utc`
- **Split:** 808,618 comments for training and 202,155 held out for testing, saved as `Train_Balanced.csv` and `Test_Balanced.csv`
- **Engineered features:** the CNN notebooks read `train_bal_2.csv`, which adds features such as word count, capital letters and punctuation

## Approaches

| Notebook | What it does |
|---|---|
| `DS_440_Test` | Exploratory analysis, preprocessing, Word2Vec embeddings, t-SNE visualization, an early bidirectional LSTM |
| `DS_440_Traditional_Models` | TF-IDF logistic regression baseline, stacking with the parent comment, exploring non-text features |
| `DS_440_Traditional_Models_Logit` | Logistic regression with text and non-text features, subreddit TF-IDF, tuning of `C` |
| `RNN` | A baseline recurrent network on Word2Vec embeddings |
| `GloVe` | Logistic regression and LSTM experiments with GloVe 6B 100d and 840B 300d embeddings |
| `CNN` | CNN and CNN-BiLSTM architecture experiments with Keras Tuner |
| `CNN_Test_Accuracy` | Retrains the best CNN-BiLSTM on all training data and evaluates it on the test set |
| `BERT` | BERT internals: attention visualization with `bertviz`, masked language modeling, next-sentence prediction |
| `DistilBert` | `distilbert-base-cased` sequence classification, with continued training and early stopping |

### CNN-BiLSTM

Separate inputs take the comment and its parent comment, passing through a shared embedding layer initialized from Word2Vec. Convolutional layers feed a bidirectional LSTM, then dense layers and a sigmoid output. Variants add a second input for numeric features, and attention over the numeric features and then the text.

Keras Tuner `RandomSearch` (5 trials per run) searched these settings:

| Setting | Range |
|---|---|
| Convolution filters | First layer 32 to 128; second layer up to 64 or 128, depending on the variant |
| Convolution kernel sizes | Tuned for each layer |
| LSTM units | 64 to 256 |
| Dense units | 64 to 128 |
| Learning rate | 0.01, 0.001, 0.0001 |
| Dropout | Tuned in the later variants |

### DistilBERT

Each example joins both comments as `parent_comment: {text} child_comment: {text}`, with an 80/20 split, batch size 32 and weight decay 0.05. Training ran for one epoch, then continued with early stopping.

## Results

| Model | Evaluated on | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| DistilBERT | Test | **77.2%** | | | | |
| CNN-BiLSTM, text with parent comment | Test | 72.7% | 0.772 | 0.643 | 0.702 | 0.805 |
| Logistic regression, tuned | Test | 72.6% | 0.737 | 0.702 | 0.719 | 0.726 |
| CNN-BiLSTM with attention | Validation | 72.6% | 0.724 | 0.735 | 0.729 | 0.806 |
| Baseline RNN | Validation | 58.8% | | | | |

DistilBERT performed best, and it reached 81.5% on its validation split, so its held-out test score is the fairer figure. The CNN-BiLSTM's accuracy was close to tuned logistic regression, but its ROC AUC was much higher. The attention row is a validation result, so it shouldn't be compared directly with the test rows.

The saved GloVe outputs don't include a final score, so GloVe isn't listed. These figures come from the original notebook runs. Outputs are cleared here to keep the notebooks small, and the original repository keeps them.

## Setup

The notebooks were written for Google Colab with a GPU, and there's no pinned requirements file. Versions may need adjusting.

```sh
pip install pandas numpy scikit-learn nltk gensim spacy tensorflow keras-tuner transformers datasets bertviz wordcloud
```

1. Get the dataset, and put the CSV files in Google Drive under `MyDrive/Sarcasm_Data/`. The notebooks expect `Train_Balanced.csv` and `Test_Balanced.csv`, and the CNN notebooks also need `train_bal_2.csv` and a saved `word2vec.model`.
2. Open a notebook in Colab, and mount Drive when prompted.
3. Run the cells in order. `DS_440_Test` covers the preprocessing and embeddings, so it's the best place to start.

Pushing a model to Hugging Face needs your own token, shown in `DistilBert` as `<YOUR_HUGGING_FACE_TOKEN>`.

## Contributors

**Team:** Brian Ellis, Lindsey Rich and Elena Kern.

**Lindsey Rich** designed and tuned the CNN and CNN-BiLSTM models: the multi-input architecture using parent comments, the non-text feature input, the attention variants, the Keras Tuner search, and the final test evaluation. She also worked on the shared exploratory analysis and preprocessing.

**The wider team** developed the traditional baselines, the RNN and GloVe experiments, and the BERT and DistilBERT models, including the DistilBERT model published on Hugging Face as [`brianellis1997/distilbert-sarcascm-classifier`](https://huggingface.co/brianellis1997/distilbert-sarcascm-classifier).

## License

MIT, © 2024 Brian Ellis. See [`LICENSE`](LICENSE).
