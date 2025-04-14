import os
import glob
import re
import string
import numpy as np
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker

#Dowloading the stop words for preprocessing
nltk.download('stopwords')

input_directory = "input_data"
csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
df_list = [pd.read_csv(file) for file in csv_files]
final_df = pd.concat(df_list, ignore_index=True)
print(f"Final DataFrame shape: {final_df.shape}")


output_directory = "artifacts"
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "merged_data.csv")
final_df.to_csv(output_file, index=False)
print(f"Merged data saved to: {output_file}")