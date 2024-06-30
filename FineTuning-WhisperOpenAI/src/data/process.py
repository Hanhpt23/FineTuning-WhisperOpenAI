import json
from datasets import Dataset, DatasetDict, Audio
import argparse
import os

def load_file_json(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        list: List of JSON objects.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def extract_data_from_json(json_data):
    """
    Extract audio paths, text, and durations from JSON data.

    Args:
        json_data (list): List of JSON objects representing text data.

    Returns:
        tuple: Tuple containing audio paths, text, and durations.
    """
    audio_paths, texts, durations = [], [], []
    for entry in json_data:
        # Prepend "Dataset/" to each audio path
        full_path = os.path.join("my_data/GerMed", entry['path'])
        if os.path.exists(full_path):
            audio_paths.append(full_path)
            texts.append(entry['text'])
            durations.append(entry['duration'])
        else:
            print(f"Warning: File {full_path} not found. Skipping this entry.")
    return audio_paths, texts, durations

def filter_non_empty_audio(dataset):
    """
    Filter out entries with empty audio arrays.

    Args:
        dataset (Dataset): Dataset to be filtered.

    Returns:
        Dataset: Filtered dataset.
    """
    def has_audio_array(example):
        return example['audio']['array'].size > 0
    return dataset.filter(has_audio_array)

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process audio and text data.")
    parser.add_argument("--train_scription_path", type=str, help="Path to the train transcript file.")
    parser.add_argument("--test_scription_path", type=str, help="Path to the test transcript file.")
    parser.add_argument("--val_scription_path", type=str, help="Path to the val transcript file.")
    parser.add_argument("--push_to_hub", default=True, help="Enable if want to push dataset to the hub.")
    parser.add_argument("--name_push", type=str, help="Name of the dataset to push to the hub.")
    return parser.parse_args()

if __name__ == "__main__":
    # Usage example
    args = parse_args()

    text_train = load_file_json(args.train_scription_path)
    text_val = load_file_json(args.val_scription_path)
    text_test = load_file_json(args.test_scription_path)

    data_audio_train, text_train, durations_train = extract_data_from_json(text_train)
    data_audio_val, text_val, durations_val = extract_data_from_json(text_val)
    data_audio_test, text_test, durations_test = extract_data_from_json(text_test)

    data = DatasetDict()
    data["train"] = Dataset.from_dict({"audio": data_audio_train, "text": text_train, "duration": durations_train}).cast_column("audio", Audio())
    data["val"] = Dataset.from_dict({"audio": data_audio_val, "text": text_val, "duration": durations_val}).cast_column("audio", Audio())
    data["test"] = Dataset.from_dict({"audio": data_audio_test, "text": text_test, "duration": durations_test}).cast_column("audio", Audio())
    
    # Filter out entries with empty audio arrays
    data["train"] = filter_non_empty_audio(data["train"])
    data["val"] = filter_non_empty_audio(data["val"])
    data["test"] = filter_non_empty_audio(data["test"])


    # for i in range(len(data["test"])):
    #     print(data["test"][i])
        # print(data["train"][i])
        # print(data["val"][i])

    if args.push_to_hub:
        data.push_to_hub(args.name_push, private=True)