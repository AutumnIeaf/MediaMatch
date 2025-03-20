import pandas as pd
import numpy as np

# Read the datasets
books_df = pd.read_csv('books_summary.csv')
movies_df = pd.read_csv('imdb_top_1000.csv')
games_df = pd.read_csv('steam_app_data.csv')

# Remove duplicates from books
unique_books = books_df.drop_duplicates(subset=['book_name'])

# Use all unique books, all movies, and all games
books_sample = unique_books  # All unique books
movies_sample = movies_df    # All movies
games_sample = games_df      # All games

# Add a 'media_type' column to each dataset
books_sample['media_type'] = 'book'
movies_sample['media_type'] = 'movie'
games_sample['media_type'] = 'game'

# Select and rename relevant columns for books and movies
books_final = books_sample[['book_name', 'summaries', 'categories', 'media_type']].rename(
    columns={'book_name': 'title', 'summaries': 'description', 'categories': 'genre'}
)

movies_final = movies_sample[['Series_Title', 'Overview', 'Genre', 'media_type']].rename(
    columns={'Series_Title': 'title', 'Overview': 'description', 'Genre': 'genre'}
)

# For games, keep all original columns plus add the standardized ones
games_final = games_sample.copy()
games_final['title'] = games_sample['name']
games_final['description'] = games_sample['about_the_game']
games_final['genre'] = games_sample['genres']

# Combine all datasets using only the common columns for joining
common_columns = ['title', 'description', 'genre', 'media_type']
combined_df = pd.concat([
    books_final[common_columns], 
    movies_final[common_columns], 
    games_final[common_columns + [col for col in games_final.columns if col not in common_columns]]
], ignore_index=True)

# Save the combined dataset
combined_df.to_csv('mediamatch_subset_10mb.csv', index=False)

file_size_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Combined dataset created with {len(combined_df)} entries")
print(f"- Books: {len(books_final)} entries")
print(f"- Movies: {len(movies_final)} entries")
print(f"- Games: {len(games_final)} entries")
print(f"Estimated file size: {file_size_mb:.2f} MB")