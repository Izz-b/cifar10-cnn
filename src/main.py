# m

import torch
from train import train_and_evaluate_model
from model import CNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = CNN().to(device)
    
    # Train the model
    train_and_evaluate_model(model, device)

if __name__ == "__main__":
    main()
