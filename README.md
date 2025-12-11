# Next Word Prediction (LSTM + Streamlit)

Smartphone users expect keyboards to guess the next word instantly. This project trains an LSTM language model on a SwiftKey-style Twitter corpus and exposes it through a Streamlit app so you can type a sentence fragment and see top next-word suggestions in real time.

## Project Structure

```
.
├── NextWordPrediction.ipynb     # full training & EDA notebook
├── en_US.twitter.txt            # (ignored in git) raw SwiftKey tweets
├── next_word_model.h5           # trained Keras model (saved after training)
├── tokenizer.pkl                # fitted Keras tokenizer
├── streamlit_app.py             # Streamlit UI for inference
└── README.md
```

## Dataset

- Source: `en_US.twitter.txt` from the SwiftKey capstone corpus (tweets, short informal text).
- For memory reasons only 3,000 lines are loaded in the notebook; adjust `MAX_LINES` for larger samples.
- Dataset file is >100 MB, so it’s not committed; download it separately and place it in the project root.

## Training Pipeline (Notebook)

1. **Cleaning**: ASCII-only text (emoji removal), lowercasing, blank-line drop.
2. **Tokenizer**: Keras `Tokenizer(num_words=50_000, lower=True)` fitted on the sampled tweets.
3. **Sequence generation**: incremental n-grams from each sentence, padded to `max_length = 34`.
4. **Model**: `Embedding(7385, 100) → LSTM(150) → Dense(softmax)`.
5. **Training**: categorical cross-entropy, Adam optimizer, 40 epochs (~92% accuracy).
6. **Artifacts**: saved via
   ```python
   model.save("next_word_model.h5")
   with open("tokenizer.pkl", "wb") as f:
       pickle.dump(tokenizer, f)
   ```

## Streamlit App

`streamlit_app.py` loads the saved model/tokenizer and exposes a simple UI:

- Text input for the seed sentence.
- Slider for number of suggestions (top-1..top-5).
- “Predict” button shows the most probable next words with confidence scores.

### Running locally

```bash
pip install -r requirements.txt      # list: streamlit, tensorflow, numpy
streamlit run streamlit_app.py
```

Make sure `next_word_model.h5` and `tokenizer.pkl` are in the same folder (and set `MODEL_PATH`, `TOKENIZER_PATH`, `MAX_SEQUENCE_LEN` in the script if your names differ). The app launches at `http://localhost:8501`.

### Deployment (Streamlit Community Cloud)

1. Push this repo (minus the large dataset) to GitHub.
2. Go to https://share.streamlit.io, choose “New app”, point to the repo + `streamlit_app.py`.
3. Streamlit installs dependencies from `requirements.txt` and hosts the app under a free `.streamlit.app` URL.

## How to Recreate / Extend

- Swap in larger corpora (blogs/news) by increasing `MAX_LINES` and retraining.
- Use `sparse_categorical_crossentropy` if one-hot labels become too large.
- Experiment with hyperparameters (embedding dimension, LSTM units, dropout) via Keras Tuner.
- Add temperature/beam-search sampling for richer suggestions.
- Deploy behind an API if you prefer FastAPI/Flask over Streamlit.
