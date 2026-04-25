import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing
from collections import Counter

# For NLP
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# For ML Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





df["resume_length"] = df["Cleaned_Resume"].apply(lambda x: len(x.split()))
df["jd_length"] = df["Cleaned_JD"].apply(lambda x: len(x.split()))

plt.figure(figsize=(12,5))
sns.histplot(df["resume_length"], bins=30, kde=True, color="blue", label="Resume Length")
sns.histplot(df["jd_length"], bins=30, kde=True, color="orange", label="Job Description Length")
plt.legend()
plt.title("Distribution of Resume vs Job Description Lengths")
plt.show()

# 6.2 Top 20 most common words in resumes
all_resume_text = " ".join(df["Cleaned_Resume"].astype(str).tolist())
resume_counts = Counter(all_resume_text.split()).most_common(20)

words, counts = zip(*resume_counts)
plt.figure(figsize=(10,6))
sns.barplot(x=list(counts), y=list(words), palette="magma")
plt.title("Top 20 Most Common Words in Resumes")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()

# 6.3 Top 20 most common words in job descriptions
all_jd_text = " ".join(df["Cleaned_JD"].astype(str).tolist())
jd_counts = Counter(all_jd_text.split()).most_common(20)

words, counts = zip(*jd_counts)
plt.figure(figsize=(10,6))
sns.barplot(x=list(counts), y=list(words), palette="viridis")
plt.title("Top 20 Most Common Words in Job Descriptions")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()

# 6.4 Match score distribution
plt.figure(figsize=(10,5))
sns.histplot(df["match_score"], bins=30, kde=True, color="green")
plt.title("Distribution of Resume-Job Match Scores")
plt.xlabel("Match Score")
plt.ylabel("Frequency")
plt.show()
