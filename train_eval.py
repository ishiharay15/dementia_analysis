import torch
import csv
from tqdm import tqdm
from .models import CrossAttention
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device, save_name, patience=5):
    model = model.to(device)

    best_val_loss = float("inf")
    best_model_path = f"{save_name}.pth"

    train_losses = []
    val_losses = []
    epochs_no_improve = 0  # Counter for early stopping

    with open(f"{save_name}.csv", mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # ========== Training ==========
        for audio_embeddings, text_embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # labels = labels.to(device).float().unsqueeze(1)
            if audio_embeddings:
                audio_embeddings = audio_embeddings.to(device)
            if text_embeddings:
                text_embeddings = text_embeddings.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            if isinstance(model, CrossAttention):
                if audio_embeddings is None or text_embeddings is None:
                    continue
                outputs = model(audio_embeddings, text_embeddings)
            else:
                inputs = audio_embeddings if audio_embeddings is not None else text_embeddings
                outputs = model(inputs)

            optimizer.zero_grad()
            outputs = model(audio_embeddings, text_embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio_embeddings, text_embeddings, labels in val_loader:
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(audio_embeddings, text_embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs. Best Val Loss: {best_val_loss:.4f}")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        with open(f"{save_name}.csv", mode="a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch + 1, train_loss, val_loss])

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {epochs_no_improve}/{patience}")

    # Plot Learning Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curve")
    plt.grid()
    plt.savefig(f"{save_name}.png")
    plt.close()

    model.load_state_dict(torch.load(best_model_path))
    return model


def evaluate(model, test_loader, args, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audio_embeddings, text_embeddings, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            labels = labels.to(device).float().unsqueeze(1)
            if isinstance(model, CrossAttention):
                if audio_embeddings is None or text_embeddings is None:
                    continue
                outputs = model(audio_embeddings, text_embeddings)
            else:
                inputs = audio_embeddings if audio_embeddings is not None else text_embeddings
                outputs = model(inputs)
            preds = (outputs > 0).long().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=1)
    
    # Compute Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # Compute Specificity, Sensitivity, UAR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    uar = (specificity + sensitivity) / 2

    print("\n🔹 **Evaluation Metrics** 🔹")
    print(f"Acc: {accuracy:.4f}")
    print(f"F-1: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"UAR: {uar:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=1))

    #  Save Results to CSV
    results_file = "results/text_taukadial_results.csv"
    os.makedirs("results", exist_ok=True)

    # Check if file exists to write header only once
    write_header = not os.path.exists(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header only if the file is newly created
        if write_header:
            writer.writerow([
                "Modality", "Features", "Classifier", "Language", "Augmentations",
                "Acc", "F-1", "Sensitivity", "Specificity", "UAR"
            ])
        
        # Write results
        writer.writerow([
            "text", args.features, args.model, args.lang, 'no',
            round(accuracy, 4), round(f1, 4), round(sensitivity, 4), round(specificity, 4), round(uar, 4)
        ])