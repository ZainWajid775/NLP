import pandas as pd

# Loading dataset into dataframe
df = pd.read_csv('IMDB Dataset.csv')

# DATA PREPROCESSING AND CLEANING
import re                                   # Regular expressions for text manipulation
import nltk                                 # Natural Language Toolkit for text processing
from nltk.corpus import stopwords           # Stopwords for removing common words
from nltk.tokenize import word_tokenize     # Tokenization of text into words
from nltk.stem import WordNetLemmatizer     # Lemmatization for reducing words to their base form

nltk.download('stopwords')                  # Filter non-informative words
nltk.download('punkt')                      # Tokenization of text
nltk.download('wordnet')                    # WordNet for lemmatization (base words)

# Function for text cleaning
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)                                             # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                                         # Remove non-alphabetic characters
    text = text.lower()                                                             # Convert to lowercase
    tokens = word_tokenize(text)                                                    # Tokenize text into words
    tokens = [word for word in tokens if word not in stopwords.words('english')]    # Remove stopwords
    lemmatizer = WordNetLemmatizer()                                                # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]                        # Lemmatize words
    return ' '.join(tokens)                                                         # Join tokens back into a single string

# Apply text cleaning to the 'review' column
from tqdm import tqdm
tqdm.pandas()
df['cleaned_review'] = df['review'].progress_apply(clean_text)

# FEATURE EXTRACTION 
# Assign weights to each word and use them to predict the sentiment of the review
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)             # Use top 5000 words
X = vectorizer.fit_transform(df['cleaned_review'])          # Assign weights
y = df['sentiment'].map({'positive': 1, 'negative': 0})     # Map sentiment to binary values


# MODEL TRAINING
from sklearn.model_selection import train_test_split                # For splitting the dataset
from sklearn.linear_model import LogisticRegression                 # Logistic Regression model (Binary Classification)
from sklearn.metrics import accuracy_score, classification_report   # For evaluating the model 

# Split the dataset into training and testing sets (20 test, 80 train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# SAVE MODEL AND VECTORISE
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'sentiment_model.pkl')

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import os
print("Files in current directory:", os.listdir())