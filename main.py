import os
import sys

import nltk
import glob
import pandas as pd
from src.predictor.utils import chat_words_dict
from src.predictor.components.preprocessing import TextPreprocessor
from src.predictor.logger import logging
from src.predictor.exception import SentimentException

#Dowloading the stop words for preprocessing
nltk.download('stopwords')

# Load all CSVs
input_directory = "input_data"
csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
df_list = [pd.read_csv(file) for file in csv_files]
final_df = pd.concat(df_list, ignore_index=True)

# Save merged file
os.makedirs("artifacts", exist_ok=True)
final_df.to_csv("artifacts/merged_data.csv", index=False)

# Merge and prepare reviews
final_df['review'] = final_df['title'] + " " + final_df['description']
review = final_df[['review']].drop_duplicates().reset_index(drop=True).dropna()

# Apply preprocessing

try :
    logging.info("Preprocessing Launched")
    preprocessor = TextPreprocessor(chat_words_dict)
    review['processed_reviews'] = review['review'].apply(preprocessor.preprocess)

except Exception as e:
            raise SentimentException(e,sys)

# Save processed data
review.to_csv("artifacts/preprocessed_data.csv", index=False)
print(f'Data preprocessing successful with {review.shape} shape of data')
