from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean resumes
df["combined_text"] = df["Cleaned_Resume"] + " " + df["Cleaned_JD"]

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1500)

X_tfidf = tfidf_vectorizer.fit_transform(df["combined_text"])
y = df["match_score"]  # or whichever column represents the category/score

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Shape of TF-IDF matrix:", X_tfidf.shape)
print("Sample TF-IDF features:", tfidf_vectorizer.get_feature_names_out()[:10])
