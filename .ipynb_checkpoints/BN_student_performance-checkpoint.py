#!/usr/bin/python3
#!/usr/bin/env python3
"""
BN_student_performance.py
---------------------------------
Probabilistic Graphical Modeling of Student Performance Using Bayesian Networks

Author: Thomas Kitaba
Course: Probabilistic and Graphical Models (AAU, MSc in AI)
Dataset: UCI "Student Performance" (student-mat.csv, student-por.csv)

Objective:
-----------
Analyze factors influencing student academic performance
using a Bayesian Network (BN). The model captures dependencies
among study time, failures, family support, prior grades, etc.,
and predicts final grades under uncertainty.

Requirements:
-------------
pip install pandas numpy matplotlib seaborn pgmpy networkx requests
"""

# === 1. Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import zipfile
import io
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings

warnings.filterwarnings("ignore")

# === 2. Download and extract dataset function ===
def download_and_extract_dataset(url, extract_to="."):
    """
    Download and unzip a file from a URL
    """
    print(f"⬇️ Downloading dataset from {url} ...")
    r = requests.get(url)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=extract_to)
        print("✅ Dataset downloaded and extracted successfully.")
    else:
        raise Exception(f"Failed to download dataset: Status code {r.status_code}")

# Dataset URL (UCI Student Performance)
dataset_url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

# Download only if the files do not exist
if not (os.path.exists("student-mat.csv") and os.path.exists("student-por.csv")):
    download_and_extract_dataset(dataset_url)
else:
    print("✅ Dataset files already exist. Skipping download.")

# === 3. Load and merge datasets ===
mat = pd.read_csv("student-mat.csv", sep=";")
por = pd.read_csv("student-por.csv", sep=";")

data = pd.concat([mat, por], axis=0).reset_index(drop=True)
print("✅ Dataset Loaded and Merged")
print("Shape:", data.shape)

# === 4. Preprocessing ===
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

# Select subset of relevant columns for simplicity
cols = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
    'Medu', 'Fedu', 'studytime', 'failures', 'schoolsup',
    'famsup', 'goout', 'health', 'absences', 'G1', 'G2', 'G3'
]
data = data[cols]

print("\n✅ Sample Data:")
print(data.head())

# === 5. Exploratory Visualization ===
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Selected Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("✅ Saved correlation heatmap → correlation_heatmap.png")

# === 6. Learn Bayesian Network Structure ===
print("\n🔍 Learning Bayesian Network structure using Hill Climb Search + BIC...")
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BIC(data))  # Pass scoring here

# Use DiscreteBayesianNetwork for latest pgmpy
model = DiscreteBayesianNetwork(best_model.edges())
print("✅ Learned Structure:")
print(model.edges())

# === 7. Learn CPDs (Maximum Likelihood Estimation) ===
model.fit(data, estimator=MaximumLikelihoodEstimator)
print("\n✅ Model fitted with Maximum Likelihood Estimation")

# === 8. Visualize Bayesian Network ===
plt.figure(figsize=(12, 8))
G = nx.DiGraph(model.edges())
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10, arrows=True)
plt.title("Learned Bayesian Network Structure")
plt.tight_layout()
plt.savefig("bayesian_network_structure.png")
plt.close()
print("✅ Saved Bayesian Network structure → bayesian_network_structure.png")

# === 9. Inference Example ===
inference = VariableElimination(model)
query_result = inference.map_query(variables=['G3'], evidence={'studytime': 2, 'failures': 1})
print(f"\n🎯 Predicted G3 (final grade) given studytime=2, failures=1 → {query_result['G3']}")

# === 10. Save learned structure ===
edges_df = pd.DataFrame(list(model.edges()), columns=['Parent', 'Child'])
edges_df.to_csv("learned_structure.csv", index=False)
print("✅ Saved learned structure → learned_structure.csv")

# === 11. Project Summary ===
print("\n📊 Project Summary:")
print("- Dataset shape:", data.shape)
print("- Number of variables used:", len(cols))
print("- Number of edges learned:", len(model.edges()))
print("- Example inference output:", query_result)
print("\n✅ All tasks completed successfully!")
