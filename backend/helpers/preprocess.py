import pandas as pd
import numpy as np
import pickle
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')


def load_and_clean_data():
    # Load datasets
    print("Loading datasets...")
    books_df = pd.read_csv('data/books_summary.csv')
    movies_df = pd.read_csv('data/imdb_top_1000.csv')
    games_df = pd.read_csv('data/steam_app_data.csv')

    # Remove duplicates from books dataset
    print("Removing duplicates from books dataset...")
    books_df = books_df.drop_duplicates(subset=['book_name'])

    # Select and rename relevant columns
    books_clean = books_df[['book_name', 'summaries', 'categories']].rename(
        columns={'book_name': 'title',
                 'summaries': 'description', 'categories': 'genre'}
    )
    books_clean['media_type'] = 'book'

    movies_clean = movies_df[['Series_Title', 'Overview', 'Genre']].rename(
        columns={'Series_Title': 'title',
                 'Overview': 'description', 'Genre': 'genre'}
    )
    movies_clean['media_type'] = 'movie'

    # For games, use about_the_game or detailed_description
    games_df['description'] = games_df['about_the_game'].fillna(
        games_df['detailed_description'])
    # Clean up genres (assuming they're in a string format that needs parsing)
    games_clean = games_df[['name', 'description', 'genres']].rename(
        columns={'name': 'title'}
    )
    games_clean['media_type'] = 'game'

    # Combine all datasets
    print("Combining datasets...")
    combined_df = pd.concat(
        [books_clean, movies_clean, games_clean], ignore_index=True)

    # Fill missing values
    combined_df['description'] = combined_df['description'].fillna('')
    combined_df['genre'] = combined_df['genre'].fillna('')

    # Filter out entries with empty descriptions
    combined_df = combined_df[combined_df['description'].str.len() > 5]

    print(f"Total unique media items: {len(combined_df)}")
    return combined_df


def preprocess_text(text):
    """Tokenize, remove stopwords, and stem the text"""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and stem remaining words
    filtered_tokens = [stemmer.stem(
        word) for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(filtered_tokens)


def create_tfidf_embeddings(df):
    print("Creating TF-IDF embeddings for all descriptions...")

    # Create a combined text field with title and description
    df['combined_text'] = df['title'] + ". " + df['description']

    # Preprocess the text
    print("Preprocessing text...")
    df['processed_text'] = df['combined_text'].progress_apply(preprocess_text)

    # Create and fit TF-IDF vectorizer
    print("Fitting TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit to top 1000 features
        min_df=5,           # Ignore terms that appear in fewer than 5 documents
        max_df=0.85,        # Ignore terms that appear in more than 85% of documents
        sublinear_tf=True   # Apply sublinear tf scaling (1 + log(tf))
    )

    # Create the TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

    # Convert sparse matrix to list of arrays for storage
    print("Converting TF-IDF matrix to arrays...")
    embeddings = []
    for i in tqdm(range(tfidf_matrix.shape[0])):
        embeddings.append(tfidf_matrix[i].toarray()[0])

    # Save embeddings
    print("Saving embeddings...")
    df['embedding'] = embeddings

    # Save the vectorizer and processed data
    with open('models/tfidf_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'df': df,
            'vectorizer': tfidf_vectorizer,
            'feature_names': tfidf_vectorizer.get_feature_names_out()
        }, f)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return df, tfidf_vectorizer


if __name__ == "__main__":
    # Apply tqdm to pandas operations
    tqdm.pandas()

    # Process data
    combined_df = load_and_clean_data()
    combined_df, vectorizer = create_tfidf_embeddings(combined_df)
    print("Preprocessing complete!")
