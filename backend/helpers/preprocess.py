import pandas as pd
import numpy as np
import pickle
import nltk
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_and_clean_data():
    with open('data/books.json', 'r') as f:
        books_data = json.load(f)
    with open('data/movies.json', 'r') as f:
        movies_data = json.load(f)
    with open('data/games.json', 'r') as f:
        games_data = json.load(f)
    
    books_df = pd.DataFrame(books_data)
    movies_df = pd.DataFrame(movies_data)
    games_df = pd.DataFrame(games_data)
    
    books_df = books_df.drop_duplicates(subset=['book_name'])
    
    books_clean = books_df[['book_name', 'summaries', 'categories']].rename(
        columns={'book_name': 'title', 'summaries': 'description', 'categories': 'genre'}
    )
    books_clean['media_type'] = 'book'
    
    movies_clean = movies_df[['Series_Title', 'Overview', 'Genre']].rename(
        columns={'Series_Title': 'title', 'Overview': 'description', 'Genre': 'genre'}
    )
    movies_clean['media_type'] = 'movie'
    
    games_df['description'] = games_df['about_the_game'].fillna(games_df['detailed_description'])
    games_clean = games_df[['name', 'description', 'genres']].rename(
        columns={'name': 'title'}
    )
    games_clean['media_type'] = 'game'
    
    combined_df = pd.concat([books_clean, movies_clean, games_clean], ignore_index=True)
    
    combined_df['description'] = combined_df['description'].fillna('')
    combined_df['genre'] = combined_df['genre'].fillna('')
    
    combined_df = combined_df[combined_df['description'].str.len() > 5]
    
    return combined_df

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def create_tfidf_embeddings(df):
    df['combined_text'] = df['title'] + ". " + df['description']
    df['processed_text'] = [preprocess_text(text) for text in tqdm(df['combined_text'])]
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.85,
        sublinear_tf=True
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    
    embeddings = []
    for i in tqdm(range(tfidf_matrix.shape[0])):
        embeddings.append(tfidf_matrix[i].toarray()[0])
    
    df['embedding'] = embeddings
    
    with open('models/tfidf_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'df': df,
            'vectorizer': tfidf_vectorizer,
            'feature_names': tfidf_vectorizer.get_feature_names_out()
        }, f)
    
    return df, tfidf_vectorizer

def load_from_combined_json():
    with open('data/mediamatch_combined.json', 'r') as f:
        combined_data = json.load(f)
    
    combined_df = pd.DataFrame(combined_data)
    combined_df['description'] = combined_df['description'].fillna('')
    combined_df['genre'] = combined_df['genre'].fillna('')
    combined_df = combined_df[combined_df['description'].str.len() > 5]
    
    return combined_df

if __name__ == "__main__":
    use_combined_json = False
    
    if use_combined_json:
        combined_df = load_from_combined_json()
    else:
        combined_df = load_and_clean_data()
    
    combined_df, vectorizer = create_tfidf_embeddings(combined_df)