import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "admin"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(
    LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(
        word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)


with open('./helpers/models/tfidf_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    df = data['df']
    vectorizer = data['vectorizer']

embeddings = np.stack(df['embedding'].values)


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    # default all
    media_type = request.form.get('media_type', 'all')

    if not query:
        return jsonify({'error': 'No query provided'})

    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query]).toarray()
    cos_scores = cosine_similarity(query_vector, embeddings)[0]

    if media_type != 'all':
        media_mask = df['media_type'] == media_type
        filtered_scores = [score if mask else 0 for score,
                           mask in zip(cos_scores, media_mask)]
        cos_scores = filtered_scores

    top_results = np.argsort(-np.array(cos_scores))[:10]

    results = []
    for idx in top_results:
        results.append({
            'title': df.iloc[idx]['title'],
            'media_type': df.iloc[idx]['media_type'],
            'genre': df.iloc[idx]['genre'],
            'description': df.iloc[idx]['description'],
            'score': float(cos_scores[idx])
        })

    return jsonify({'results': results})


@app.route('/explain', methods=['GET'])
def explain_recommendation():
    item_id = request.args.get('id')
    query = request.args.get('query')

    if not item_id or not query:
        return jsonify({'error': 'Missing item_id or query parameter'})

    item_id = int(item_id)
    item = df.iloc[item_id]

    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query]).toarray()[0]
    item_vector = embeddings[item_id]

    feature_names = vectorizer.get_feature_names_out()
    term_contributions = query_vector * item_vector

    top_term_indices = np.argsort(-term_contributions)[:5]
    top_terms = [(feature_names[idx], float(term_contributions[idx]))
                 for idx in top_term_indices if term_contributions[idx] > 0]

    explanation = {
        'title': item['title'],
        'media_type': item['media_type'],
        'matching_terms': top_terms
    }

    return jsonify(explanation)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
