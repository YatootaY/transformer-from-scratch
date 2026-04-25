import torch
import torch.nn as nn
import torch.optim as optim
import time

from data import get_dataloaders
from model import MiniTransformer
from utils import save_training_curve, set_seed

def train_and_evaluate():
    seed = 67
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    curve_csv = "train_curve.csv"
    curve_png = "train_curve.png"

    set_seed(seed)
    print(f"seed {seed}")
    print(f"batch_size {batch_size} epochs {epochs} lr {learning_rate}")
    
    print("loading data")
    train_loader, val_loader, test_loader = get_dataloaders(
        'data/train.csv', 'data/validation.csv', 'data/test.csv', batch_size=batch_size
    )
    
    print("initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MiniTransformer(
        vocab_size=5, 
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=1,
        use_pe=True
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"training on {device}")
    start_time = time.time()
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for tokens, masks, labels in train_loader:
            tokens, masks, labels = tokens.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(tokens, masks)
            
            loss = criterion(predictions, labels)
            
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for tokens, masks, labels in val_loader:
                tokens, masks, labels = tokens.to(device), masks.to(device), labels.to(device)
                
                logits = model(tokens, masks)
                
                probs = torch.sigmoid(logits)
                
                preds = (probs > 0.5).float()
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_accuracy = correct / total

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_accuracy": round(val_accuracy * 100, 4),
            }
        )
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}%")

    training_time = (time.time() - start_time) / 60
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("done")
    print(f"time_min {training_time:.2f}")
    print(f"params {total_params}")

    plot_path = save_training_curve(history, csv_path=curve_csv, plot_path=curve_png, title="Training Curve")
    print(f"curve_csv {curve_csv}")
    if plot_path is not None:
        print(f"curve_plot {plot_path}")
    else:
        print("curve_plot skipped")

if __name__ == "__main__":
    train_and_evaluate()