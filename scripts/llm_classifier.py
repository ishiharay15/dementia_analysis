import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, recall_score
from huggingface_hub import login
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Define supported LLMs

LLAMA_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

SUPPORTED_LLMS = {
    "llama": LLAMA_MODELS[0],
    "flan-t5": "google/flan-t5-large",
    "mistral": "mistralai/Mistral-7B-Instruct",
    "falcon": "tiiuae/falcon-7b-instruct",  # NEW
    "gpt-neox": "EleutherAI/gpt-neox-20b",  # NEW
    "opt": "facebook/opt-6.7b"  # NEW
}


# Argument parser with defaults
parser = argparse.ArgumentParser(description="Dementia Classification using LLMs")
parser.add_argument("--llm", type=str, choices=SUPPORTED_LLMS.keys(), required=True, help="Select the LLM model to use")
parser.add_argument("--csv", type=str, default="TAUKADIAL-24/test.csv", help="Path to the test dataset CSV")
parser.add_argument("--lang", type=str, choices=["en", "cn", "all"], default="all", help="Choose language: 'en' (English), 'zh' (Chinese), or 'all'")
parser.add_argument("--hf_token", type=str, help="Hugging Face token for authentication")
args = parser.parse_args()

# # Prompt format

### LLAMA
PROMPT_TEMPLATE = """Classify the cognitive status of the speaker based on the provided text. Assess the description for linguistic signs of cognitive impairment, such as difficulty finding words, simplified syntax, repetition, and reduced content. Respond strictly with either NC (Normal Cognition) or MCI (Mild Cognitive Impairment). 

Text: "{text}"

Answer:"""

## T5
# PROMPT_TEMPLATE = """Determine if the following text indicates NC (Normal Cognition) or MCI (Mild Cognitive Impairment).
# Analyze language patterns such as word retrieval difficulty, repetition, disfluency, and reduced content.
# Respond with exactly one word: "NC" or "MCI".

# Text: "{text}"

# Answer:"""

# Mapping labels
LABEL_MAP = {"NC": 0, "MCI": 1}

# Custom Dataset with Language Filtering
class DementiaDataset(Dataset):
    def __init__(self, csv_file, lang="en"):
        self.data = pd.read_csv(csv_file)

        # Filter dataset based on language
        if lang == "en":
            self.data = self.data[self.data["lang"] == "eng_Latn"]
        elif lang == "cn":
            self.data = self.data[self.data["lang"].isin(["zho_Hant", "yue_Hant"])]
        elif lang == "all":
            pass  # Use all rows

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_path = row["text"]  # Path to .txt file

        # Read the actual text from the file
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()  # Read and remove extra spaces/newlines
        except FileNotFoundError:
            print(f"Warning: File {text_path} not found. Skipping entry.")
            return None  # Skip entry if file is missing

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return text, label


# Function to initialize LLaMA model
def initialize_llama(hf_token, llama_id):
    """
    Authenticate and load LLaMA model.
    """
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(llama_id)
    model = AutoModelForCausalLM.from_pretrained(llama_id, torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding

    return tokenizer, model



# Function to run inference using LLaMA
# Function to run inference using LLaMA
def run_llama_inference(test_loader, hf_token, llama_id):
    """
    Runs LLaMA inference on test data and returns predictions and labels.
    """
    # Load LLaMA model
    tokenizer, model = initialize_llama(hf_token, llama_id)

    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  # Extract text
        prompt = PROMPT_TEMPLATE.format(text=text)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.0, do_sample=False)

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # print(f"!!!! Generated Response: {response}")  # Debugging line

        # Extract only "NC" or "MCI"
        extracted_label = extract_nc_mci(response)

        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response!")  # Debugging line

    return y_true, y_pred

def initialize_flan_t5(model_id):
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

# Function to run inference using FLAN-T5
def run_flan_t5_inference(test_loader, model_id):
    """
    Runs Flan-T5 inference on test data and returns predictions and labels.
    """
    tokenizer, model = initialize_flan_t5(model_id)
    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  # Extract text
        prompt = f"Does the following text indicate NC (Normal Cognition) or MCI (Mild Cognitive Impairment)? Answer in one word: NC or MCI.\n\nText: \"{text}\"\n\nAnswer:"

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # Limit length to force short response
            temperature=0.7,  # Add randomness to avoid overconfidence
            repetition_penalty=1.2,
            do_sample=True
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Generated Response: {response}")  # Debugging

        # Extract only "NC" or "MCI"
        extracted_label = extract_nc_mci(response)

        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response! {response}")  # Debugging

    return y_true, y_pred


def initialize_mistral(model_id, hf_token):
    """
    Load Mistral-7B-Instruct model.
    """
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return tokenizer, model

def run_mistral_inference(test_loader, model_id, hf_token):
    """
    Runs Mistral-7B-Instruct inference on test data and returns predictions and labels.
    """
    tokenizer, model = initialize_mistral(model_id, hf_token)
    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  # Extract text
        prompt = f"Classify the following text into NC (Normal Cognition) or MCI (Mild Cognitive Impairment). \n\nText: \"{text}\"\n\nFinal Answer:"

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=10,  # Keep response short
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            do_sample=False
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Generated Response: {response}")  # Debugging

        # Extract only "NC" or "MCI"
        extracted_label = extract_nc_mci(response)

        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response! {response}")  # Debugging

    return y_true, y_pred


def initialize_falcon(model_id, hf_token):
    """
    Load Falcon model.
    """
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return tokenizer, model

def run_falcon_inference(test_loader, model_id, hf_token):
    """
    Runs Falcon inference on test data and returns predictions and labels.
    """
    tokenizer, model = initialize_falcon(model_id, hf_token)
    model_device = next(model.parameters()).device  # Get model's actual device

    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  # Extract text
        prompt = f"Classify the following text into NC (Normal Cognition) or MCI (Mild Cognitive Impairment).\n\nText: \"{text}\"\n\nFinal Answer:"

        # Tokenize and send to model's device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}  # Move inputs to correct device

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            repetition_penalty=1.2,
            do_sample=True
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Generated Response: {response}")  # Debugging

        # Extract only "NC" or "MCI"
        extracted_label = extract_nc_mci(response)

        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response! {response}")  # Debugging

    return y_true, y_pred



def initialize_gpt_neox(model_id, hf_token):
    """
    Load GPT-NeoX model.
    """
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return tokenizer, model

def run_gpt_neox_inference(test_loader, model_id, hf_token):
    """
    Runs GPT-NeoX inference on test data.
    """
    tokenizer, model = initialize_gpt_neox(model_id, hf_token)
    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  
        prompt = f"Given the following text, determine if the speaker has NC (Normal Cognition) or MCI (Mild Cognitive Impairment).\n\nText: \"{text}\"\n\nFinal Answer:"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model.generate(**inputs, max_length=10, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, do_sample=False)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Generated Response: {response}")

        extracted_label = extract_nc_mci(response)
        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response! {response}")

    return y_true, y_pred


def initialize_opt(model_id, hf_token):
    """
    Load Meta's OPT model.
    """
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    return tokenizer, model

def run_opt_inference(test_loader, model_id, hf_token):
    """
    Runs OPT inference on test data.
    """
    tokenizer, model = initialize_opt(model_id, hf_token)
    y_true, y_pred = [], []

    for text, true_label in test_loader:
        text = text[0]  
        prompt = f"Based on the given text, classify it as NC (Normal Cognition) or MCI (Mild Cognitive Impairment).\n\nText: \"{text}\"\n\nFinal Answer:"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model.generate(**inputs, max_length=10, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.2, do_sample=False)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Generated Response: {response}")

        extracted_label = extract_nc_mci(response)
        if extracted_label in LABEL_MAP:
            y_true.append(true_label.item())
            y_pred.append(LABEL_MAP[extracted_label])
        else:
            print(f">>>>>> Invalid Response! {response}")

    return y_true, y_pred


# Function to extract NC/MCI from response
def extract_nc_mci(response):
    """
    Extracts NC or MCI from the model response.
    """
    response = response.split("Answer:")[-1].strip()  # Extract text after "Answer:"
    
    if response.startswith("NC"):
        return "NC"
    elif response.startswith("MCI"):
        return "MCI"
    
    return None  # If neither NC nor MCI is found



# Placeholder function for other LLMs
def run_other_llm_inference(test_loader, hf_token, model_id):
    """
    Placeholder for other LLM inference functions.
    """
    raise NotImplementedError(f"Inference for {model_id} is not implemented yet.")


# Load test dataset with language filter
test_dataset = DementiaDataset(args.csv, lang=args.lang)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if args.llm == "llama":
    y_true, y_pred = run_llama_inference(test_loader, args.hf_token, SUPPORTED_LLMS["llama"])
elif args.llm == "flan-t5":
    y_true, y_pred = run_flan_t5_inference(test_loader, SUPPORTED_LLMS["flan-t5"])
elif args.llm == "mistral":
    y_true, y_pred = run_mistral_inference(test_loader, SUPPORTED_LLMS["mistral"], args.hf_token)
elif args.llm == "falcon":
    y_true, y_pred = run_falcon_inference(test_loader, SUPPORTED_LLMS["falcon"], args.hf_token)
elif args.llm == "gpt-neox":
    y_true, y_pred = run_gpt_neox_inference(test_loader, SUPPORTED_LLMS["gpt-neox"], args.hf_token)
elif args.llm == "opt":
    y_true, y_pred = run_opt_inference(test_loader, SUPPORTED_LLMS["opt"], args.hf_token)
else:
    y_true, y_pred = run_other_llm_inference(test_loader, args.hf_token, SUPPORTED_LLMS[args.llm])



# Run evaluation
if not y_pred:
    print("No valid predictions were made. Check LLM responses.")
else:
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    uar = recall_score(y_true, y_pred, average="macro")

    print(f"Model: {args.llm}")
    print(f"Language: {args.lang}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"UAR: {uar:.4f}")
