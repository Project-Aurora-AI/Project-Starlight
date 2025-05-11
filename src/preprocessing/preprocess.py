import os
import json
import logging
import random
import pandas as pd
import nltk
from tqdm import tqdm
from src.utils.file_io import save_preprocessed_data, load_preprocessed_data
from .cleaning import clean_text
from .tokenizer import tokenize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set NLTK dependencies (if needed for advanced tokenization)
nltk.download('punkt')

def preprocess_data(input_path, output_path, augment=False):
    """
    Main preprocessing pipeline to clean, tokenize, and save the data.
    
    Args:
        input_path (str): Path to the raw input data.
        output_path (str): Path to save the processed data.
        augment (bool): Whether to apply data augmentation.
    """
    logger.info(f"Loading raw data from {input_path}...")
    raw_data = load_raw_data(input_path)

    logger.info("Cleaning the data...")
    cleaned_data = [clean_text(item) for item in tqdm(raw_data)]

    if augment:
        logger.info("Augmenting the data...")
        cleaned_data = augment_data(cleaned_data)

    logger.info("Tokenizing the data...")
    tokenized_data = [tokenize_text(item) for item in tqdm(cleaned_data)]

    logger.info(f"Saving processed data to {output_path}...")
    save_preprocessed_data(tokenized_data, output_path)

def load_raw_data(file_path):
    """
    Load raw data from a file. Supports both JSON and CSV formats.
    
    Args:
        file_path (str): Path to the raw data.
    
    Returns:
        list: Raw data in a list format.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path).to_dict(orient='records')
    else:
        raise ValueError(f"Unsupported file type for {file_path}")

def augment_data(data):
    """
    Augment the data (e.g., by paraphrasing, random insertion, etc.).
    
    Args:
        data (list): List of cleaned data entries.
    
    Returns:
        list: Augmented data.
    """
    augmented_data = []
    
    for item in data:
        # Example of a simple augmentation (random sampling)
        if random.random() > 0.9:  # 10% chance to augment
            augmented_data.append(paraphrase(item))
        else:
            augmented_data.append(item)
    
    return augmented_data

def paraphrase(text):
    """
    Generate a paraphrased version of the input text.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Paraphrased text.
    """
    # This could be a call to an external model or a library like Hugging Face transformers
    # For now, a simple random word swap is implemented as an example.
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

def process_in_chunks(file_path, chunk_size=1000, augment=False):
    """
    Process large datasets in chunks to avoid memory issues.
    
    Args:
        file_path (str): Path to the raw data.
        chunk_size (int): Number of entries to process at once.
        augment (bool): Whether to apply data augmentation.
    """
    logger.info(f"Processing data in chunks of size {chunk_size}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(line.strip())
            
            if len(chunk) >= chunk_size:
                logger.info(f"Processing chunk {i // chunk_size + 1}...")
                preprocess_data_chunk(chunk, augment)
                chunk = []  # Reset chunk
            
        # Process the last chunk if necessary
        if chunk:
            preprocess_data_chunk(chunk, augment)

def preprocess_data_chunk(data_chunk, augment):
    """
    Preprocess a chunk of data (called during chunk processing).
    
    Args:
        data_chunk (list): List of data entries to process.
        augment (bool): Whether to apply data augmentation.
    """
    # Clean, augment, and tokenize the chunk
    cleaned_data = [clean_text(item) for item in data_chunk]
    if augment:
        cleaned_data = augment_data(cleaned_data)
    tokenized_data = [tokenize_text(item) for item in cleaned_data]
    
    # Save or further process the tokenized chunk here
    logger.info(f"Processed {len(tokenized_data)} entries.")

