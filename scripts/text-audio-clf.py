import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models import *
from dataset import AudioTextDataset, collate_fn
from train_eval import train, evaluate
from transformers import BertTokenizer, BertModel
import torchaudio.pipelines as pipelines

def get_encoders(args, device):
    tokenizer, text_encoder, audio_encoder = None, None, None

    if "BERT" in args.feature_set:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased").to(device).eval()

    if "wave2vec" in args.feature_set:
        audio_encoder = pipelines.WAV2VEC2_BASE.get_model().to(device).eval()

    return tokenizer, text_encoder, audio_encoder

def get_savepath(args):
    save_name = f'{"-".join(args.feature_set)}_{args.model}_{args.optimizer}_{args.scheduler}_{args.lang}'
    save_dir = f'text_audio_results/{save_name}'
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Results will be saved in: {save_dir}")
    save_path = f'{save_dir}/{save_name}'
    return save_path

def optim_scheduler(model, args):
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None

    return optimizer, scheduler

def get_dataloaders(args, device):
    tokenizer, text_encoder, audio_encoder = get_encoders(args, device)
    train_dataset = AudioTextDataset(args.train_csv, args.model, args.feature_set, tokenizer, text_encoder, audio_encoder, args.lang)
    test_dataset = AudioTextDataset(args.test_csv, args.model, args.feature_set, tokenizer, text_encoder, audio_encoder, args.lang)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def get_model(args):
    input_dim_audio = sum([768 if feat in args.feature_set else 0 for feat in ["wave2vec", "egemaps", "librosa"]])
    input_dim_text = 768 if "BERT" in args.feature_set else 0

    if input_dim_audio > 0 and input_dim_text > 0:
        model_dict = {
            "mlp": MLPClassifier(input_dim_audio + input_dim_text, use_attention=args.use_attention),
            "lstm": LSTMMLPClassifier(input_dim_audio + input_dim_text),
            "cross_atten": CrossAttention(input_dim_audio, input_dim_text),
        }
    elif input_dim_audio > 0:  # Audio-only 
        model_dict = {
            "mlp": MLPClassifier(input_dim_audio, use_attention=args.use_attention),
            "lstm": LSTMMLPClassifier(input_dim_audio),
        }
    elif input_dim_text > 0:  # Text-only
        model_dict = {
            "mlp": MLPClassifier(input_dim_text, use_attention=args.use_attention),
            "lstm": LSTMMLPClassifier(input_dim_text),
        }
    else:
        raise ValueError("At least one feature set must be selected.")

    return model_dict[args.model]

def main(args):
    torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}'

    # Dataloaders
    train_loader, test_loader = get_dataloaders(args)

    # Model selection
    model = get_model(args).to(device)

    # Optimizer, scheduler, loss
    optimizer, scheduler = optim_scheduler(model, args)
    loss_fn = nn.BCEWithLogitsLoss()

    # Saving
    save_path = get_savepath(args)

    # Train
    print(f"Training `{args.model}` with `{args.audio_features}` & `{args.text_features}` features on `{args.lang}` data...")
    model = train(model, train_loader, test_loader, args.epochs, loss_fn, optimizer, scheduler, device, save_path)

    # Evaluate
    evaluate(model, test_loader, args, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="TAUKADIAL-24/train.csv")
    parser.add_argument("--test_csv", type=str, default="TAUKADIAL-24/test.csv")
    parser.add_argument("--model", type=str, choices=["mlp", "lstm", "cross_atten"], default="mlp")
    parser.add_argument("--use_attention", action="store_true", help="Use self-attention in MLP classifiers")
    parser.add_argument("--feature_set", nargs="+", choices=["wave2vec", "wavbert", "egemaps", "librosa", "BERT"], default=["wave2vec", "BERT"])
    parser.add_argument("--lang", type=str, choices=["eng", "chin", "all"], default="eng")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "plateau"], default="cosine")

    args = parser.parse_args()
    main(args)



