
# Step 1: Set Up the Environment
# Run this in your terminal if not already installed:
# pip install pandas numpy matplotlib seaborn nltk scikit-learn streamlit wordcloud

# Step 2: Load the Dataset
import pandas as pd

df = pd.read_csv("chatgpt_reviews.csv")
print(df.head())

# Step 3: Data Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

# Step 4: Label Sentiments
def get_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(get_sentiment)

# Step 5: Exploratory Data Analysis (Example Word Cloud)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

positive_words = " ".join(df[df['sentiment'] == 'Positive']['clean_review'])
wordcloud = WordCloud(width=800, height=400).generate(positive_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Step 6: Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review']).toarray()
y = df['sentiment']

# Step 7: Model Training
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
