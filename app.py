import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, \
    add_prediction_details, view_all_prediction_details, create_emotionclf_table, VN  # Import IST from track_utils

model = load_model('models/ckpt4.h5')

with open('models/tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

labels = ['sadness', 'joy', 'neutral', 'anger', 'fear', 'surprise']
maxlen = 97

emotions_emoji_dict = {
    "sadness": "😔",
    "joy": "😂",
    "neutral": "😔",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮"
}


def predict_emotions(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

    predictions = model.predict(padded_sequences)

    predicted_label = labels[np.argmax(predictions)]
    return predicted_label


def get_prediction_proba(docx):
    sequences = tokenizer.texts_to_sequences([docx])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

    predictions = model.predict(padded_sequences)
    return predictions

def main():
    st.title("Emotion Classifier App")

    create_page_visited_table()
    create_emotionclf_table()

    add_page_visited_details("Home", datetime.now(VN))
    st.subheader("Emotion Detection in Text")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(VN))

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "😐")
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence: {:.2f}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=labels)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
