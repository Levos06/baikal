import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

import torch
import yaml
from data_utils import create_dataloaders, BaikalDataset, NoiseSigGraphPreprocessor
from models import load_model
from metrics import binary_clf_metrics

def smoke_test():
    config_path = "src/train_configs/gcn_3layer_512hs.yaml"
    with open(config_path, "r") as f:
        train_params = yaml.safe_load(f)

    print(f"Loading data from: {train_params['path_to_data']}")
    
    # Check if file exists
    if not os.path.exists(train_params['path_to_data']):
        print(f"ERROR: Data file not found at {train_params['path_to_data']}")
        # Try /home2 instead of /home3 just in case
        alt_path = train_params['path_to_data'].replace("home3", "home2")
        print(f"Trying alternative path: {alt_path}")
        if os.path.exists(alt_path):
            print("Found it!")
            train_params['path_to_data'] = alt_path
        else:
            print("Alternative path also failed. Searching for any .h5 files...")
            return

    # Initialize preprocessor
    preprocessor = NoiseSigGraphPreprocessor(train_params["knn_neighbours"])
    
    # Create dataloaders
    try:
        dataloaders = create_dataloaders(
            path_to_data=train_params["path_to_data"],
            is_graph=train_params["is_graph"],
            batch_size=train_params["batch_size"],
            DatasetType=BaikalDataset,
            is_classification=True,
            preprocessor=preprocessor,
            num_workers=0 # For smoke test
        )
        print("Dataloaders created successfully")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # Load model
    model = load_model(train_params["model_type"], train_params["model_params"])
    print("Model loaded successfully")
    print(model)

    # Test one batch
    train_loader = dataloaders["train"]
    batch = next(iter(train_loader))
    print(f"Batch type: {type(batch)}")
    
    if train_params["is_graph"]:
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        y = batch.y
    else:
        x = batch[0]
        y = batch[1]
        edge_index = None
        batch_idx = None
    
    print(f"Input shape: {x.shape}")
    if edge_index is not None:
        print(f"Edge index shape: {edge_index.shape}")
    
    output = model(x, edge_index, batch_idx)
    print(f"Output shape: {output.shape}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, y)
    print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    smoke_test()
