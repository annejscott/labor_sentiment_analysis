import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials from environment variables
twitter_api_key = os.getenv("TWITTER_API_KEY")
twitter_api_secret = os.getenv("TWITTER_API_SECRET")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_secret = os.getenv("REDDIT_SECRET")


# Verify credentials are loaded
if not twitter_api_key or not twitter_api_secret:
    print(
        "Warning: Twitter API credentials are missing. Please add them to the .env file."
    )
else:
    print("Twitter API credentials loaded successfully!")

# Additional verification for other setup aspects
import sys
import pandas as pd
import numpy as np
from transformers import pipeline

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print("Environment setup is working!")
