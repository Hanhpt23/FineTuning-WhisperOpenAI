import argparse
import logging
import sys
from pprint import pprint
from datasets import load_dataset, Audio, DatasetDict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
    )
import argparse
import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from evaluate import load

import torch
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union





def load_data_and_model(data_name, sub_name, model_name, num_proc, split="test"):
    # Load dataset
    data = load_dataset(data_name, sub_name, split=split, num_proc=num_proc)
    
    # Load model and related components
    model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model_input_name = feature_extractor.model_input_names[0]
    processor = WhisperProcessor.from_pretrained(model_name)
    normalizer = BasicTextNormalizer()
    
    # Cast audio column to appropriate format
    data = data.cast_column("audio", Audio(sampling_rate=16000))
    
    return data, model, feature_extractor, model_input_name, processor, normalizer

def tokenizer_data(data, feature_extractor, model_input_name, normalizer, text_column_name):
    # Extract audio features
    audio = data["audio"]
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    data[model_input_name] = input_features.get(model_input_name)[0]
    
    # Normalize and prepare reference text
    input_str = data[text_column_name].lower()
    input_str = normalizer(input_str).strip()
    data["reference"] = input_str
    
    return data

def map_to_pred(batch, model, processor, normalizer, model_input_name, language):
    # Generate predictions
    with torch.no_grad():
        predicted_ids = model.generate(batch[model_input_name].to(torch.device('cuda')), language=language)
    
    # Decode predictions
    pred_str = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    result = [normalizer(pred).strip() for pred in pred_str]
    
    return {"prediction": result}

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Whisper models.')
    # parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to load.')
    # parser.add_argument('--model_name', type=str, required=True, help='Name of the model to load.')
    # parser.add_argument('--language', type=str, required=True, help='Language of the data.')
    parser.add_argument('--num_proc', type=int, default=2, help='Number of processes to use for data loading.')
    return parser.parse_args()

def filter_empty_strings(references, predictions):
    filtered_references = []
    filtered_predictions = []
    for ref, pred in zip(references, predictions):
        if ref and pred:
            filtered_references.append(ref)
            filtered_predictions.append(pred)
    return filtered_references, filtered_predictions

def main(args):
    label_name = "text"
    languages = ['vi', 'en', 'fr', 'zh', 'de']
    # split data for validation
    splits = ['dev', 'eval', 'eval', 'eval', 'val']
    # languages = ['en']
    # data_names = ['pphuc25/EngMed']
    data_names = ['pphuc25/VietMed-split-8-2', 'pphuc25/EngMed', 'pphuc25/FranceMed', 'pphuc25/ChiMed', 'Hanhpt23/GermanMed-full']
    model_names = {
        'vi': ['Hanhpt23/whisper-medium-Encod-vietmed', 'Hanhpt23/whisper-small-Encod-vietmed', 'Hanhpt23/whisper-base-Encod-vietmed', 'Hanhpt23/whisper-tiny-Encod-vietmed'],
        'en': ['Hanhpt23/whisper-medium-Encode-engmed', 'Hanhpt23/whisper-small-Encode-engmed', 'Hanhpt23/whisper-base-Encode-engmed', 'Hanhpt23/whisper-tiny-Encode-engmed'],
        'fr': ['Hanhpt23/whisper-medium-Encod-frenchmed', 'Hanhpt23/whisper-small-Encod-frenchmed', 'Hanhpt23/whisper-base-Encod-frenchmed', 'Hanhpt23/whisper-tiny-Encod-frenchmed'],
        'zh': ['Hanhpt23/whisper-medium-Encode-chinesemed', 'Hanhpt23/whisper-small-Encode-chinesemed', 'Hanhpt23/whisper-base-Encode-chinesemed', 'Hanhpt23/whisper-tiny-Encode-chinesemed'],
        'de': ['Hanhpt23/whisper-medium-Encode-GermanMed-full', 'Hanhpt23/whisper-small-Encode-GermanMed-full', 'Hanhpt23/whisper-base-Encode-GermanMed-full', 'Hanhpt23/whisper-tiny-Encode-GermanMed-full']
    }

    batch_size = 8
    sub_name = "default"
    wer_metric = load("wer")
    cer_metric = evaluate.load("cer")
    num_proc = args.num_proc

    for i, language in enumerate(languages):
        data_name = data_names[i]
        split = splits[i]
        model_name_list = model_names[language]

        for model_name in model_name_list:
            try:
                data, model, feature_extractor, model_input_name, processor, normalizer = load_data_and_model(data_name, sub_name, model_name, num_proc, split)
            except Exception as e:
                print(f"Error loading data or model for {model_name}: {e}")
                continue

            try:
                data_tokenizer = data.map(lambda data: tokenizer_data(data, feature_extractor, model_input_name, normalizer, label_name), num_proc=num_proc).with_format("torch")
                result = data_tokenizer.map(lambda batch: map_to_pred(batch, model, processor, normalizer, model_input_name, language), batched=True, batch_size=batch_size)
                
                # Filter out empty strings
                references, predictions = filter_empty_strings(result["reference"], result["prediction"])
                
                if references and predictions:
                    wer_score = 100 * wer_metric.compute(references=references, predictions=predictions)
                    cer_score = 100 * cer_metric.compute(predictions=predictions, references=references)
                    print(f"Model: {model_name} - WER: {wer_score:.2f}, - CER: {cer_score:.2f}")
                else:
                    print(f"Model: {model_name} - No valid references or predictions for evaluation")
            except Exception as e:
                print(f"Error during tokenization or prediction for {model_name}: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
