import numpy as np
import pandas as pd
import optuna
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
import pandas as pd

# Objective function for Optuna
def objective(trial, df):
    # Hyperparameters to tune
    n_neighbors = trial.suggest_int("n_neighbors", 3, 100)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    n_components = trial.suggest_int("n_components", 2, 10)
    min_cluster_size = trial.suggest_int("min_cluster_size", 5, 50)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.5)

    # Select embedding model as a hyperparameter
    sent_model = trial.suggest_categorical("sent_model", [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ])

    # Define UMAP and HDBSCAN with suggested hyperparameters
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=epsilon)

    # Load Sentence Transformer Model
    embedding_model = SentenceTransformer(sent_model)

    # Create BERTopic model
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model)

    # Fit Model
    topics, _ = topic_model.fit_transform(df)

    # Metric: Number of unique topics (more balanced topic distribution is better)
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)  # Ignore outliers (-1)

    return num_topics  # Maximize number of detected topics

# Function to run Optuna optimization and save best params to CSV
def optimize_topic_model(df):
    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df), n_trials=30)

    # Save best hyperparameters to CSV
    best_params = study.best_params
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv('artifacts/best_params.csv', index=False)

    # Return best hyperparameters
    return best_params
