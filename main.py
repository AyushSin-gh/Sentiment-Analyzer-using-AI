import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import speech_recognition as sr
import re

# Set the Streamlit page configuration FIRST
st.set_page_config(page_title="Advanced Sentiment Analyzer", layout="wide", page_icon="💡")

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Load the accuracy from file (as percentage) and convert to decimal
@st.cache_resource
def load_accuracy():
    try:
        with open("model_accuracy.txt", "r") as f:
            accuracy_value = float(f.read().strip())
            if accuracy_value > 1:  # If the value is greater than 1, it's already a percentage
                accuracy_decimal = accuracy_value / 100  # Convert percentage to decimal
            else:
                accuracy_decimal = accuracy_value  # Already in decimal format
            return accuracy_decimal
    except Exception as e:
        st.error(f"Error reading accuracy: {e}")
        return None

model, vectorizer = load_model()
accuracy = load_accuracy()

# Streamlit App Interface
st.title("📝 Advanced Sentiment Analysis Tool")

# Show the accuracy in decimal format (up to 2 decimal places)
if accuracy is not None:
    st.write(f"Model Accuracy: {accuracy:.2f}")  # Display as decimal with 2 decimal places
else:
    st.warning("Accuracy could not be loaded.")

# Add Navbar
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Choose an option", ["Text Analysis ✍", "Voice Input 🎤", "CSV Analysis 📄", "WhatsApp Chat Analysis 💬"])

if nav_option == "Text Analysis ✍":
    # User Input for text sentiment analysis
    user_input = st.text_area("Enter a sentence to analyze sentiment:")

    if st.button("Analyze Sentiment"):
        if user_input:
            # Vectorize the input and predict sentiment
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

            # Map predictions to sentiment
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😢" if prediction == -1 else "Neutral 😐"
            st.success(f"The predicted sentiment is: **{sentiment}**")
        else:
            st.warning("Please enter some text to analyze.")
    
    st.write("💡 This tool uses a pre-trained model to analyze text sentiment.")

if nav_option == "Voice Input 🎤":
    st.subheader("🎤 Voice Recognition for Sentiment Analysis")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    if st.button("Start Recording"):
        with mic as source:
            st.info("Listening... Speak now.")
            audio = recognizer.listen(source, timeout=10)
            try:
                text_from_audio = recognizer.recognize_google(audio)
                st.write(f"Recognized Text: {text_from_audio}")
                input_vector = vectorizer.transform([text_from_audio])
                sentiment = model.predict(input_vector)[0]
                sentiment_type = "Positive 😊" if sentiment == 1 else "Negative 😢" if sentiment == -1 else "Neutral 😐"
                st.write(f"Predicted Sentiment: {sentiment_type}")
            except Exception as e:
                st.error(f"Error recognizing speech: {e}")
elif nav_option == "CSV Analysis 📄":
    st.subheader("📄 Analyze CSV File")
    upl = st.file_uploader("Upload a CSV file", type=['csv'])
    if upl:
        try:
            df = pd.read_csv(upl)
            if 'text' not in df.columns:
                st.error("The CSV file must contain a 'text' column.")
            else:
                df['Polarity'] = df['text'].apply(lambda x: model.predict(vectorizer.transform([x]))[0])
                df['Sentiment'] = df['Polarity'].apply(lambda x: 'Positive 😊' if x == 1 else 'Negative 😢' if x == -1 else 'Neutral 😐')

                st.write(df.head())
                st.bar_chart(df['Sentiment'].value_counts())

                st.download_button("Download Processed CSV", df.to_csv(index=False), file_name="processed_sentiment.csv")
        except Exception as e:
            st.error(f"Error: {e}")

elif nav_option == "WhatsApp Chat Analysis 💬":
    st.subheader("💬 Analyze WhatsApp Chat File")
    upl = st.file_uploader("Upload WhatsApp Chat File (.txt)", type=['txt'])
    if upl:
        try:
            chat_data = upl.getvalue().decode("utf-8")
            messages = re.findall(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?: AM| PM)?) - ([^:]+): (.+)", chat_data)
            chat_df = pd.DataFrame(messages, columns=["Datetime", "Sender", "Message"])

            chat_df['Polarity'] = chat_df['Message'].apply(lambda x: model.predict(vectorizer.transform([x]))[0])
            chat_df['Sentiment'] = chat_df['Polarity'].apply(lambda x: 'Positive 😊' if x == 1 else 'Negative 😢' if x == -1 else 'Neutral 😐')

            st.bar_chart(chat_df['Sentiment'].value_counts())
            st.download_button("Download Processed Chat", chat_df.to_csv(index=False), file_name="processed_chat.csv")
        except Exception as e:
            st.error(f"Error parsing chat: {e}")

# Footer
st.markdown("<footer><b>Advanced Sentiment Analyzer</b> | Developed by Team Fault </footer>", unsafe_allow_html=True)
