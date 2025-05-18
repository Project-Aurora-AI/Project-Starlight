from datasets import load_dataset

dataset1 = load_dataset('rajpurkar/squad', split='train')  # Training split of IMDb dataset

# Load external datasets
ultra_fineweb = load_dataset("openbmb/Ultra-FineWeb", split="train[:10%]")
ag_news = load_dataset("fancyzhx/ag_news", split="train[:10%]")

# Combine datasets
combined_datasets = []
combined_datasets.extend([item["text"] for item in ultra_fineweb])
combined_datasets.extend([item["text"] for item in ag_news])

# Example: Print the number of samples in each dataset
print(f"Number of Ultra-FineWeb samples: {len(ultra_fineweb)}")
print(f"Number of AG News samples: {len(ag_news)}")
print(f"Total combined samples: {len(combined_datasets)}")
