import torch
import torch.nn as nn
import torch.optim as optim
import time

from data import get_dataloaders
from model import MiniTransformer
from utils import save_training_curve, set_seed, slugify

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, masks, labels in dataloader:
            tokens, masks, labels = tokens.to(device), masks.to(device), labels.to(device)
            logits = model(tokens, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100


def run_experiment(config, train_loader, val_loader, test_loader, device, epochs, learning_rate, output_dir):
    config_name = config["name"]
    use_pe = config["use_pe"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    seed = config["seed"]

    set_seed(seed)

    print(f"config {config_name}")
    print(f"seed {seed} use_pe {use_pe} heads {num_heads} layers {num_layers}")
    
    model = MiniTransformer(
        vocab_size=5, 
        d_model=64, 
        num_heads=num_heads, 
        d_ff=128,
        num_layers=num_layers,
        dropout_rate=0.1,
        use_pe=use_pe
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    start_time = time.time()
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for tokens, masks, labels in train_loader:
            tokens, masks, labels = tokens.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_accuracy": round(val_acc, 4),
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
            
    train_time = (time.time() - start_time) / 60
    
    val_acc = history[-1]["val_accuracy"]
    test_acc = evaluate_model(model, test_loader, device)

    base_name = slugify(config_name)
    curve_csv = f"{output_dir}/{base_name}_curve.csv"
    curve_plot_path = f"{output_dir}/{base_name}_curve.png"
    curve_plot = save_training_curve(
        history,
        csv_path=curve_csv,
        plot_path=curve_plot_path,
        title=f"Training Curve: {config_name}",
    )
    
    print(f"done time_min {train_time:.2f} val {val_acc:.1f} test {test_acc:.1f}")
    print(f"curve_csv {curve_csv}")
    if curve_plot is not None:
        print(f"curve_plot {curve_plot}")
    else:
        print("curve_plot skipped")
    
    return {
        "Config": config_name,
        "Params": total_params,
        "Time (m)": round(train_time, 2),
        "Val Acc (%)": round(val_acc, 2),
        "Test Acc (%)": round(test_acc, 2),
        "Curve CSV": curve_csv,
        "Curve Plot": curve_plot if curve_plot is not None else "N/A",
    }


def main():
    seed = 67
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    output_dir = "benchmark_outputs"

    set_seed(seed)

    print("loading data")
    train_loader, val_loader, test_loader = get_dataloaders(
        'data/train.csv', 'data/validation.csv', 'data/test.csv', batch_size=batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    print(f"batch_size {batch_size} epochs {epochs} lr {learning_rate}")

    if not torch.jit.is_scripting():
        import os
        os.makedirs(output_dir, exist_ok=True)

    configs = [
        {"name": "Base (PE=True, Heads=4, Layers=1)", "use_pe": True, "num_heads": 4, "num_layers": 1, "seed": seed},
        {"name": "No PE (PE=False, Heads=4, Layers=1)", "use_pe": False, "num_heads": 4, "num_layers": 1, "seed": seed},
        {"name": "Single Head (PE=True, Heads=1, Layers=1)", "use_pe": True, "num_heads": 1, "num_layers": 1, "seed": seed},
        {"name": "Two Layers (PE=True, Heads=4, Layers=2)", "use_pe": True, "num_heads": 4, "num_layers": 2, "seed": seed},
    ]
    
    results = []
    
    for cfg in configs:
        res = run_experiment(
            cfg,
            train_loader,
            val_loader,
            test_loader,
            device,
            epochs,
            learning_rate,
            output_dir,
        )
        results.append(res)
        
    print("results")
    print("config,params,time_min,val_acc,test_acc")
    for r in results:
        print(f"{r['Config']},{r['Params']},{r['Time (m)']},{r['Val Acc (%)']},{r['Test Acc (%)']}")
    print(f"output_dir {output_dir}")

if __name__ == "__main__":
    main()