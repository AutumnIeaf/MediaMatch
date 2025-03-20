import pandas as pd
import numpy as np

# Read the datasets
books_df = pd.read_csv('books_summary.csv')
movies_df = pd.read_csv('imdb_top_1000.csv')
games_df = pd.read_csv('steam_app_data.csv')


# combine all datasets into 1
# Remove duplicates from books
unique_books = books_df.drop_duplicates(subset=['book_name'])
# Use all unique books, all movies, and all games
books_sample = unique_books  # All unique books
movies_sample = movies_df    # All movies
games_sample = games_df      # All games

#makes games_sample a subset
games_sample = games_sample.sample(frac=0.7, random_state=1)

# combine all samples into 1 df
combined_df = pd.concat([books_sample, movies_sample, games_sample], ignore_index=True)
combined_df.to_csv('mediamatch_subset_10mb.csv', index=False)