from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = Path("next_word_model.h5")
TOKENIZER_PATH = Path("tokenizer.pkl")
MAX_SEQUENCE_LEN = 34


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model and tokenizer once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Save the trained model before running the app."
        )
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {TOKENIZER_PATH}. Pickle the fitted tokenizer before running the app."
        )

    model = load_model(MODEL_PATH)
    with TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_next_words(
    model, tokenizer, text: str, top_k: int = 3
) -> List[Tuple[str, float]]:
    """Return top-k next-word suggestions with probabilities."""
    sequence = tokenizer.texts_to_sequences([text.lower().strip()])
    if not sequence or not sequence[0]:
        return []

    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LEN, padding="pre")
    probs = model.predict(padded, verbose=0)[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    index_word = tokenizer.index_word

    suggestions = []
    for idx in top_indices:
        word = index_word.get(idx)
        if word:
            suggestions.append((word, float(probs[idx])))
    return suggestions


def main():
    st.set_page_config(page_title="Next Word Predictor", page_icon="⌨️")
    st.title("Next Word Prediction")
    st.caption("Powered by an LSTM trained on SwiftKey-style Twitter data")

    try:
        model, tokenizer = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    user_text = st.text_input(
        "Type a sentence prefix (at least one known word):",
        placeholder="e.g., i am so",
    )
    top_k = st.slider("Number of suggestions", min_value=1, max_value=5, value=3)

    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter a sentence fragment before predicting.")
            return

        suggestions = predict_next_words(model, tokenizer, user_text, top_k)
        if not suggestions:
            st.info("No suggestion available. Try adding more context or different words.")
            return

        st.subheader("Suggestions")
        for word, prob in suggestions:
            st.write(f"- **{word}** (confidence: {prob:.2%})")


if __name__ == "__main__":
    main()

