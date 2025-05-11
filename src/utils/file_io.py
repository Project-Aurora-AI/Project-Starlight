import os
import json
import yaml
import torch
import shutil

def save_config(config, path):
    """
    Saves the configuration dictionary to a YAML file.
    Args:
        config (dict): The configuration dictionary to save.
        path (str): The path where the config should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directories if they don't exist
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {path}")

def load_config(path):
    """
    Loads configuration from a YAML file.
    Args:
        path (str): The path to the config file.
    
    Returns:
        dict: The loaded configuration.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {path}")
    return config

def save_preprocessed_data(data, file_path):
    """
    Save preprocessed data (e.g., tokenized text) to a file.
    
    Args:
    - data (any type): Data to be saved (typically a list, dict, etc.).
    - file_path (str): Path where the data will be saved.
    
    Returns:
    - None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Make sure the directory exists
    with open(file_path, 'w') as file:
        json.dump(data, file)

def load_preprocessed_data(file_path):
    """
    Load preprocessed data (e.g., tokenized text) from a file.
    
    Args:
    - file_path (str): Path to the saved file containing the data.
    
    Returns:
    - Data loaded from the file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, path):
    """
    Saves data to a JSON file.
    Args:
        data (dict/list): The data to save.
        path (str): The path to save the data to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")

def load_json(path):
    """
    Loads data from a JSON file.
    Args:
        path (str): The path to the JSON file.
    
    Returns:
        dict/list: The loaded data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from {path}")
    return data

def save_model(model, path):
    """
    Saves the model state_dict to a specified path.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path where the model will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Loads a model state_dict from a specified path.
    Args:
        model (torch.nn.Module): The model to load the weights into.
        path (str): The path to the saved model.
    
    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"Model at {path} does not exist.")
    return model

def create_directory(path):
    """
    Creates a directory if it does not exist.
    Args:
        path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at {path}")
    else:
        print(f"Directory already exists at {path}")

def clean_directory(path):
    """
    Deletes all contents in a directory.
    Args:
        path (str): The path of the directory to clean.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Directory {path} cleaned.")
    else:
        print(f"Directory {path} does not exist.")

def move_file(src, dest):
    """
    Moves a file from source to destination.
    Args:
        src (str): The source file path.
        dest (str): The destination file path.
    """
    if os.path.exists(src):
        shutil.move(src, dest)
        print(f"Moved file from {src} to {dest}")
    else:
        print(f"Source file {src} does not exist.")

def file_exists(path):
    """
    Checks if a file or directory exists at the given path.
    Args:
        path (str): The path to check.
    
    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    return os.path.exists(path)

