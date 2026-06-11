import streamlit as st
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# DOWNLOAD NLTK DATA
# ======================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# ======================
# CUSTOM CSS (YOUR COLOR PALETTE ONLY)
# ======================
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #FBF5DD;
    }
    h1, h2, h3 {
        color: #306D29;
    }
    .story-box {
        background-color: #E7E1B1;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #0D530E;
        margin-bottom: 15px;
    }
    .movie-card {
        background-color: #E7E1B1;
        padding: 18px;
        border-radius: 14px;
        border-left: 8px solid #306D29;
        margin-bottom: 18px;
    }
    .movie-title {
        color: #0D530E;
        font-size: 20px;
        font-weight: bold;
    }
    .movie-story {
        color: #306D29;
        font-size: 15px;
    }
    textarea {
        background-color: #E7E1B1 !important;
        color: #0D530E !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("imdb_cleaned.csv")

# remove duplicates (important)
df = df.drop_duplicates(subset=["Movie Name"]).reset_index(drop=True)

# ======================
# NLP SETUP
# ======================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ======================
# TEXT CLEANING (WITH LEMMATIZATION)
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    # ✅ LEMMATIZATION ADDED HERE
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# ======================
# APPLY CLEANING
# ======================
df["clean_story"] = df["Storyline"].apply(clean_text)

# ======================
# TF-IDF VECTORIZATION
# ======================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_story"])

# ======================
# RECOMMENDATION FUNCTION
# ======================
def recommend_from_storyline(user_story, top_n=5):
    user_clean = clean_text(user_story)
    user_vector = vectorizer.transform([user_clean])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    sorted_indices = similarity_scores.argsort()[::-1]

    seen = set()
    recommendations = []

    for idx in sorted_indices:
        movie = df.iloc[idx]

        if movie["Movie Name"] not in seen:
            seen.add(movie["Movie Name"])
            recommendations.append(movie)

        if len(recommendations) == top_n:
            break

    return recommendations

# ======================
# UI HEADER
# ======================
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Find movies using NLP + TF-IDF + Cosine Similarity</h3>", unsafe_allow_html=True)

# ======================
# USER INPUT
# ======================
st.markdown("<div class='story-box'>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter a movie storyline or plot description:",
    height=150,
    placeholder="Example: A warrior fights powerful enemies in a futuristic world..."
)

st.markdown("</div>", unsafe_allow_html=True)

# ======================
# BUTTON ACTION
# ======================
if st.button("🎥 Recommend Movies"):
    if user_input.strip() == "":
        st.warning("Please enter a storyline to get recommendations.")
    else:
        results = recommend_from_storyline(user_input)

        st.markdown("<h2>🔝 Top 5 Recommended Movies</h2>", unsafe_allow_html=True)

        for i, movie in enumerate(results, 1):
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{i}. {movie['Movie Name']}</div>
                    <div class="movie-story">{movie['Storyline']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================
# FOOTER
# ======================
st.markdown(
    "<hr><p style='color:#306D29;'>Built with NLP + Lemmatization + TF-IDF + Cosine Similarity</p>",
    unsafe_allow_html=True
)