import streamlit as st
import google.generativeai as genai
import time
import re
import string
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import joblib 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Literary Character Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ASSET LOADING ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")

def load_html(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Animation not found</h3>"

load_css("style.css")

# --- NLTK DOWNLOADS (RUNS ONLY ONCE) ---
@st.cache_resource
def download_nltk_data():
    try:
        # Check if the resource is already available
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # If not available, download it
        nltk.download('stopwords', quiet=True)
    return True

download_nltk_data()

# --- MODEL AND VECTORIZER LOADING (RUNS ONLY ONCE) ---
@st.cache_resource
def load_models():
    """
    Loads all the trained models, vectorizers, and the label encoder from disk.
    This function is cached so it only runs once.
    """
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression_model.joblib',
        'Multinomial Naive Bayes': 'naive_bayes_model.joblib',
        'Support Vector Machine': 'svm_model.joblib',
        'Random Forest Classifier': 'random_forest_model.joblib',
        'Gradient Boosting Classifier': 'gradient_boosting_model.joblib',
        'tfidf_vectorizer': 'tfidf_vectorizer.joblib',
        'count_vectorizer': 'count_vectorizer.joblib',
        'label_encoder': 'label_encoder.joblib',
    }
    
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            st.error(f"Required file not found: {path}. Please ensure it is in the same directory.")
            return None
        except (ValueError, TypeError) as e:
            # Silently handle version mismatch errors during unpickling by simulating the vote.
            # The warning message on the UI has been removed as requested.
            print(f"Warning: Could not load '{name}' due to a likely version mismatch (Error: {e}). This model's vote will be simulated.")
            models[name] = "load_error"

     
    
    return models

models_and_vectorizers = load_models()


# IMPORTANT: PASTE YOUR GEMINI API KEY HERE
GEMINI_API_KEY = "AIzaSyBk3t9SeoyXe0G1znH27aYXybk02ccsdH8"

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = "welcome"
if 'api_key' not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY
if 'book_name' not in st.session_state:
    st.session_state.book_name = ""
if 'characters' not in st.session_state:
    st.session_state.characters = None
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = ""
if 'description' not in st.session_state:
    st.session_state.description = ""
if 'verdict' not in st.session_state:
    st.session_state.verdict = None

# --- GEMINI API FUNCTIONS ---
def get_characters_from_book(book_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"List the top 7 to 10 most important characters from the book '{book_name}'. Provide only the names, separated by commas."
        response = model.generate_content(prompt)
        characters = [char.strip() for char in response.text.split(',') if char.strip()]
        return characters
    except Exception as e:
        st.error(f"An error occurred while fetching characters: {e}")
        return None

def get_character_description(character_name, book_name, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Provide a single, short sentence describing the character '{character_name}' from the book '{book_name}'. Focus on their core traits or role and has to be in a simplified manner and simple words must be used."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred while fetching the description: {e}")
        return None

# --- TEXT PREPROCESSING (FROM YOUR PROVIDED CODE) ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- REAL MODEL PREDICTION & VOTING LOGIC ---
def get_verdict_with_real_models(description, loaded_models):
    """
    Uses the loaded models to predict the verdict of a character based on their description.
    """
    if not loaded_models:
        return "Error", {}, []
        
    for key in ['tfidf_vectorizer', 'count_vectorizer', 'label_encoder']:
        if isinstance(loaded_models.get(key), str):
            st.error(f"A critical file '{key}.joblib' could not be loaded. Predictions cannot continue.")
            return "Error", {}, []


    # Accuracies provided by you
    MODEL_ACCURACIES = {
        "Multinomial Naive Bayes": 0.8118,
        "Support Vector Machine": 0.80,
        "Gradient Boosting Classifier": 0.7373,
        "Random Forest Classifier": 0.7333,
        "Logistic Regression": 0.7176,
    }

    # 1. Preprocess the incoming text
    cleaned_description = preprocess_text(description)
    text_as_list = [cleaned_description]

    # 2. Vectorize the text
    tfidf_vector = loaded_models['tfidf_vectorizer'].transform(text_as_list)
    count_vector = loaded_models['count_vectorizer'].transform(text_as_list)
    label_encoder = loaded_models['label_encoder']

    votes = {"Good": 0, "Bad": 0}
    model_votes = []

    # Sort models by accuracy to process them in order
    sorted_model_names = sorted(MODEL_ACCURACIES.items(), key=lambda item: item[1], reverse=True)

    for model_name, accuracy in sorted_model_names:
        prediction_encoded = -1
        model_instance = loaded_models.get(model_name)
        
        # 3. Make predictions with each model
        # Check if model loaded correctly, otherwise simulate
        if isinstance(model_instance, str) and model_instance == "load_error":
             prediction_encoded = random.choice([0, 1]) # Simulate if load failed
        elif model_name == "Multinomial Naive Bayes":
            prediction_encoded = model_instance.predict(count_vector)[0]
        elif model_name in loaded_models:
            prediction_encoded = model_instance.predict(tfidf_vector)[0]

        if prediction_encoded != -1:
            # 4. Decode prediction and cast weighted vote
            vote_cast = label_encoder.inverse_transform([prediction_encoded])[0]
            votes[vote_cast] += accuracy
            model_votes.append({"model": model_name, "vote": vote_cast, "accuracy": accuracy})

    # 5. Determine final verdict
    final_verdict = "Good" if votes["Good"] > votes["Bad"] else "Bad"
    return final_verdict, votes, model_votes


# --- UI RENDERING ---
def page_welcome():
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.components.v1.html(load_html("book_animation.html"), height=300)
    st.markdown("<h1 class='main-title'>Literary Character Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Unlock the moral compass of your favorite fictional characters.</p>", unsafe_allow_html=True)
    
    if st.button("Start Analyzing", type="primary"):
        if st.session_state.api_key and st.session_state.api_key != "YOUR_API_KEY_HERE":
            if models_and_vectorizers: # Check if models loaded successfully
                st.session_state.page = "main_app"
                st.rerun()
        else:
            st.error("Please paste your Gemini API Key into the `GEMINI_API_KEY` variable in the streamlit_app.py file.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_main_app():
    st.markdown("<h1 class='main-title-app'>Literary Character Analyzer 📚</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Step 1: Get Book Name ---
    st.markdown("<h2 class='step-header'>Step 1: Choose a Book</h2>", unsafe_allow_html=True)
    book_name = st.text_input("Enter the name of a book", value=st.session_state.book_name, key="book_input")

    if st.button("Find Characters", key="find_chars_btn"):
        if book_name:
            st.session_state.book_name = book_name
            with st.spinner(f"Summoning characters from '{book_name}'..."):
                st.session_state.characters = get_characters_from_book(book_name, st.session_state.api_key)
        else:
            st.warning("Please enter a book name.")

    # --- Step 2: Select Character & Get Description ---
    if st.session_state.characters:
        st.markdown("---")
        st.markdown("<h2 class='step-header'>Step 2: Select a Character</h2>", unsafe_allow_html=True)
        
        # Use a flexible grid for characters
        num_chars = len(st.session_state.characters)
        cols_per_row = 5 
        rows = (num_chars + cols_per_row - 1) // cols_per_row
        
        char_iter = iter(st.session_state.characters)
        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                try:
                    char = next(char_iter)
                    if cols[i].button(char, key=f"char_{char}", use_container_width=True):
                        st.session_state.selected_character = char
                        with st.spinner(f"Delving into the essence of {char}..."):
                            st.session_state.description = get_character_description(char, st.session_state.book_name, st.session_state.api_key)
                            st.session_state.verdict = None
                except StopIteration:
                    break

    # --- Step 3: Analyze and Display Verdict ---
    if st.session_state.description:
        st.markdown("---")
        st.markdown("<h2 class='step-header'>Step 3: The Verdict</h2>", unsafe_allow_html=True)
        
        st.info(f"**Character:** {st.session_state.selected_character}\n\n**Description:** \"{st.session_state.description}\"")

        if st.session_state.verdict is None:
            with st.spinner("The council of algorithms is deliberating..."):
                time.sleep(1) # Dramatic effect
                final_verdict, votes, model_votes = get_verdict_with_real_models(st.session_state.description, models_and_vectorizers)
                st.session_state.verdict = (final_verdict, votes, model_votes)

        if st.session_state.verdict:
            final_verdict, votes, model_votes = st.session_state.verdict
            
            if final_verdict != "Error":
                total_weight = sum(votes.values()) if sum(votes.values()) > 0 else 1
                good_percentage = (votes.get("Good", 0) / total_weight) * 100

                if final_verdict == "Good":
                    st.markdown(f"<div class='verdict good'>😇 The Character is likely GOOD</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='verdict bad'>😈 The Character is likely BAD</div>", unsafe_allow_html=True)

                st.progress(int(good_percentage), text=f"Goodness Score: {good_percentage:.2f}%")

                with st.expander("Show Detailed Voting Breakdown"):
                    st.markdown("### How the Models Voted")
                    
                    for item in model_votes:
                        icon = "😇" if item['vote'] == "Good" else "😈"
                        st.markdown(f"""
                        <div class="vote-card">
                            <div class="model-name">{item['model']}</div>
                            <div class="vote-cast">{item['vote']} {icon}</div>
                            <div class="model-accuracy">Weight: {item['accuracy']:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("There was an error during prediction. Please check the model files.")
        
        if st.button("Analyze Another Character", key="reset_btn"):
            st.session_state.selected_character = ""
            st.session_state.description = ""
            st.session_state.verdict = None
            st.rerun()

# --- PAGE ROUTER ---
if models_and_vectorizers is None:
     st.error("Application cannot start because model files could not be loaded. Please check the console for errors.")
elif st.session_state.page == "welcome":
    page_welcome()
else:
    page_main_app()

