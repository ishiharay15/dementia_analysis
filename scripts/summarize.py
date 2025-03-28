from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import csv
import spacy
from collections import defaultdict

CATEGORIES = {
        1: (
            "family drive",
            {
                "father", "mother", "boy", "girl", "children", "family", "driver", "passengers",
                "car", "window", "steering", "hat", "bow", "bubble", "gum", "trees", "leaves",
                "dog", "luggage", "mirror", "sitting", "leaning", "talking", "looking", "watching",
                "chewing", "blowing", "excited", "curious", "surprised", "trip", "vacation", "summer"
            }
        ),
        2: (
            "cat rescue",
            {
                "man", "girl", "firefighters", "cat", "dog", "tree", "ladder", "tricycle", "firetruck",
                "helmet", "uniform", "branch", "bark", "animal", "leaf", "ground", "vehicle",
                "climbing", "running", "rescue", "barking", "stuck", "reaching", "watching", "jumping",
                "laughing", "waiting", "alert", "confused", "help", "child", "emergency"
            }
        ),
        3: (
            "cookie theft",
            {
                "boy", "cabinet", "children", "climbing", "cookie", "curtain", "dishes", "dish",
                "family", "faucet", "girl", "house", "jar", "mother", "plate", "reaching", "sinking",
                "sneaking", "spilling", "standing", "stealing", "stool", "thief", "tree", "washing",
                "watching", "water", "window", "overflowing"
            }
        )
    }


def summarize(tokenizer, model, input_file, output_folder):

    with open(input_file,"r") as f:
        test_file = f.read()

    inputs = tokenizer.encode(test_file, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, length_penalty=2.0, max_length=150, min_length=40, no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    output_sum_path = os.path.join(output_folder, os.path.basename(input_file))   

    with open(output_sum_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(summary)

    return summary

def check_content(nlp, summary, file_path, results):
    
    filename = os.path.basename(file_path)
    filename_ = filename.replace(".txt", "")
    parts = filename_.split("-")
    picture_id = int(parts[-1]) # 1, 2, or 3 
    patient_id = "-".join(parts[:-1])

    # score_folder = output_folder / "scores"
    # score_folder.mkdir(parents=True, exist_ok=True)
    # output_score = score_folder / f"{base_name}_score.txt"

    doc = nlp(summary)

    summary_words = set()
    for token in doc:
        if token.pos_ in {"NOUN", "VERB"}:
            summary_words.add(token.lemma_.lower())

    if picture_id in CATEGORIES:
        category, keyword_set = CATEGORIES[picture_id]

        covered = summary_words & keyword_set
        missing = keyword_set - summary_words

        results[patient_id].append(len(covered))

    else:
        print("Not an included image!")

        

def main():
    parser = argparse.ArgumentParser(description="Summarize transcriptions of TAUKADIAL dataset from Whisper transcriptions.")
    parser.add_argument("--input_folder", "-i", type=str, required=True, help="Path to the folder containing transcribed .txt files.")
    parser.add_argument("-output_folder", "-o", type=str, required=True, help="Path to the folder to save summaries.")
    
    args = parser.parse_args()

    # input_folder = "../TAUKADIAL-24/transcripts/test/"

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    nlp_model = spacy.load("en_core_web_sm")

    gt_labels = pd.read_csv("TAUKADIAL-24/test.csv")

    eng_files = set(gt_labels[gt_labels["lang"] == "eng_Latn"]["text"].tolist())

    # output_folder = Path(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)


    # input_folder = Path(args.input_folder)
    # file_paths = sorted(input_folder.glob("*.txt"), key=lambda x: x.name)

    results = defaultdict(list)

    # Iterate only over matching files
    for file_path in tqdm(eng_files):
        summary = summarize(tokenizer=tokenizer, model=model, input_file=file_path, output_folder=args.output_folder)
        check_content(nlp_model, summary, file_path, results)

    for patient in results:
        if len(results[patient]) != 3:
            print(f'Error with {patient}')
        results[patient] = sum(results[patient])

    with open("summary_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient_ID", "Score"])
        for key, value in results.items():
            writer.writerow([key, value])



if __name__ == "__main__":
    
    main()