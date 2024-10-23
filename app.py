import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved models
with open(r'C:/Amazon_Review_Sentimental_Analysis/model_buildings/Models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open(r'C:/Amazon_Review_Sentimental_Analysis/model_buildings/Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(r'C:/Amazon_Review_Sentimental_Analysis/model_buildings/Models/model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower().split()  # Convert to lowercase and split
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]  # Stemming and stopwords removal
    return ' '.join(text)

# Streamlit app
st.title('Amazon Review Sentiment Analysis')

# Input from user
review_text = st.text_area("Enter your review:", "")

if st.button('Predict Sentiment'):
    if review_text:
        # Preprocess the input review
        processed_review = preprocess_text(review_text)
        
        # Vectorize the input
        review_vector = vectorizer.transform([processed_review]).toarray()
        
        # Scale the input
        review_scaled = scaler.transform(review_vector)
        
        # Predict sentiment
        prediction = model_rf.predict(review_scaled)[0]
        
        # Display the result
        if prediction == 1:
            st.success("This review is Positive 😊")
        elif prediction == 0:
            st.info("This review is Neutral 😐")
        else:
            st.error("This review is Negative 😞")
    else:
        st.warning("Please enter a review to predict.")



