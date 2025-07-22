import sys

print(f"✅ Python version: {sys.version}\n")

# Test essential imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import flask
    from flask_cors import CORS
    import joblib
    import lightgbm
    import mlflow
    import dvc
    import nltk
    from wordcloud import WordCloud
    from googleapiclient.discovery import build
    import boto3

    print("✅ All libraries imported successfully!\n")

    # Additional runtime checks
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"MLflow version: {mlflow.__version__}")
    print(f"DVC version: {dvc.__version__}")
    print(f"NLTK version: {nltk.__version__}")
    print(f"LightGBM version: {lightgbm.__version__}")

    print("\n✅ All systems go!")

except Exception as e:
    print("❌ Something went wrong!")
    print(e)
