# Sarcasm Detection with NLP

A team project comparing traditional and deep learning approaches to detecting sarcasm in Reddit comments, using both each comment and the parent comment it replied to.

**This is collaborative work.** Team 2 (DS 440): Brian Ellis, Lindsey Rich and Elena Kern. The original repository is [brianellis1997/Sarcasm_Detection](https://github.com/brianellis1997/Sarcasm_Detection), under the MIT license kept here.

## What's here

| Notebook | Approach |
|---|---|
| `DS_440_Test` | Exploratory analysis, text preprocessing, Word2Vec embeddings, t-SNE visualization, early LSTM work |
| `DS_440_Traditional_Models`, `..._Logit` | Baseline traditional classifiers |
| `RNN`, `GloVe` | A baseline RNN, and recurrent models on GloVe embeddings |
| `CNN`, `CNN_Test_Accuracy` | Multi-input CNN and CNN-BiLSTM models, tuned with Keras Tuner (see below) |
| `BERT` | BERT internals: multi-head attention visualization, parameters, tokenization, masked language modeling, next-sentence prediction |
| `DistilBert` | DistilBERT sequence classification with continued training, published to Hugging Face |

## My contribution: CNN-BiLSTM and tuning

I (Lindsey Rich) built and tuned the CNN family of models, in `CNN` and `CNN_Test_Accuracy`:

- Separate inputs for each comment and its parent comment, sharing a Word2Vec embedding layer trained on our text
- Convolutional, then bidirectional LSTM, architectures, with and without non-text features
- Attention added for the numeric features, and then for the text as well
- Keras Tuner `RandomSearch` (5 trials per run) over filters, kernel sizes, LSTM and dense units, dropout and learning rate

| My best tuned model, CNN-BiLSTM on text | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Reserved test data | 72.7% | 0.77 | 0.64 | 0.70 | 0.80 |

The attention variants reached a similar 72.6% accuracy, F1 0.73 and ROC AUC 0.81 in the experiment notebook.

I also worked on the shared exploratory notebook, `DS_440_Test`, and made a small commit to `BERT`.

## Team results

These came from notebooks led by other team members. I didn't build these models.

| Model | Split | Accuracy | F1 | ROC AUC |
|---|---|---:|---:|---:|
| DistilBERT, the project's best model | Test | 77.2% | | |
| Best traditional model, in the `_Logit` notebook | Test | 72.6% | 0.72 | 0.73 |
| Baseline RNN | Validation | 58.8% | | |

All results come from outputs saved in the original notebooks; nothing was re-run for this repository. The team's DistilBERT model is at [brianellis1997/distilbert-sarcascm-classifier](https://huggingface.co/brianellis1997/distilbert-sarcascm-classifier).

## Commit record

Non-merge commits in the original repository. They show who committed each file, not who wrote every line.

| Notebook | Lindsey Rich | Other team members |
|---|---:|---:|
| `CNN`, `CNN_Test_Accuracy` | 2 | 0 |
| `DS_440_Test` | 5 | 10 |
| `BERT` | 1 | 5 |
| `DS_440_Traditional_Models*`, `RNN`, `GloVe`, `DistilBert` | 0 | 16 |

## Running it

The notebooks were written for Google Colab and read the data from Google Drive (`/content/drive/MyDrive/...`). The dataset isn't included, so you'd need your own copy and to update those paths. The notebooks using `transformers` need a GPU to be practical, and pushing a model needs your own Hugging Face token, shown here as `<YOUR_HUGGING_FACE_TOKEN>`.

This copy has no Git history, and notebook outputs are removed, so see the original repository for both.
