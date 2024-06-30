import os
from pydub import AudioSegment
from glob import glob
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import pysrt


def process_file(file_path, output_folder_path, name="ChineseMed"):
    file_name = os.path.basename(file_path).split(".")[0]
    audio_path = os.path.join(input_path, "Audios", f"{file_name}.mp3")

    extracted_data = extract_vtt_data(file_path)
    print(extracted_data[:5])

    output_sub_folder = f"{file_name}.json"
    
    with open(f"{os.path.join(output_folder_path, output_sub_folder)}", 'w') as file:
        json.dump(extracted_data, file, ensure_ascii=False, indent=4)
    
    milisecond_specific = get_milisecond_specific(extracted_data)

    output_sub_folder = os.path.join(output_folder_path, f"{file_name}")
    # export_audio_at_timestamps(audio_path, milisecond_specific, output_sub_folder)


def extract_vtt_data(file_path):
    """Extracts start time, end time, and transcription from a VTT file.

    Args:
        file_path: Path to the VTT file.

    Returns:
        A list of dictionaries, each containing start_time, end_time, and text.
    """
    subs = pysrt.open(file_path)  # Replace with your actual file name if it's an SRT file    

    temp_duration = 0
    current_entry = {}
    time_start = None
    text = ""
    data = []
    for sub in subs:
        hours_end, minutes_end, seconds_end, miliseconds_end = sub.end.hours, sub.end.minutes, sub.end.seconds, sub.end.milliseconds
        time_end = convert_timestamp_to_milliseconds(hours_end, minutes_end, seconds_end, miliseconds_end)
        if time_start is None:
            hours_start, minutes_start, seconds_start, miliseconds_start = sub.start.hours, sub.start.minutes, sub.start.seconds, sub.start.milliseconds
            time_start = convert_timestamp_to_milliseconds(hours_start, minutes_start, seconds_start, miliseconds_start)
        text += " " + sub.text

        temp_duration += time_end - time_start

        if temp_duration > 15000:  # Higher than 15 seconds
            current_entry["start_time"] = time_start
            current_entry["end_time"] = time_end
            current_entry["duration"] = float((round((time_end - time_start) / 1000)))
            current_entry["text"] = text.strip()
            data.append(current_entry)
            current_entry = {}  # Reset for the next entry
            temp_duration = 0
            text = ""
            time_start = None

    if temp_duration != 0:
        current_entry["start_time"] = time_start
        current_entry["end_time"] = time_end
        current_entry["duration"] = float((round((time_end - time_start) / 1000)))
        current_entry["text"] = text.strip()
        data.append(current_entry)
    return data


def get_milisecond_specific(extracted_data):
    """Extracts start time, end time, and transcription from a VTT file.

    Args:
        extracted_data: A list of dictionaries, each containing start_time, end_time, and text.

    Returns:
        A list of dictionaries, each containing start_time, end_time, and text.
    """
    milisecond_specific = []
    for entry in extracted_data:
        milisecond_specific.append((entry["start_time"], entry["end_time"]))
    return milisecond_specific


def export_audio_at_timestamps(input_file, timestamps_ms, output_file_folder):
    """Splits an audio file at multiple timestamps into multiple segments.

    Args:
        input_file: Path to the input audio file.
        timestamps_ms: List of timestamps (in milliseconds) where splits should occur.
        output_file_prefix: Prefix for the output file names.
    """
    sound = AudioSegment.from_file(input_file)

    os.makedirs(output_file_folder, exist_ok=True)
    for timestamp in timestamps_ms:
        start_time, end_time = timestamp
        part = sound[start_time:end_time]
        
        part.export(f"{output_file_folder}/{start_time}.0000_{end_time}.0000.ogg", format="ogg")


def convert_timestamp_to_milliseconds(hours, minutes, seconds, miliseconds):
    """Converts a timestamp (HH:MM:SS.SSS) to milliseconds.

    Args:
        timestamp: A string in the format HH:MM:SS.SSS.

    Returns:
        The timestamp converted to milliseconds.
    """
    total_milliseconds = ((hours * 3600 + minutes * 60 + seconds) * 1000) + miliseconds
    return int(total_milliseconds)



if __name__ == "__main__":
    input_path = "/content/drive/MyDrive/dataset/EngMed_raw/Nervous"
    output_folder_path = "/content/drive/MyDrive/dataset/EngMed"
    
    # Get list of file paths
    file_text_paths = glob(f"{input_path}/*.vtt")

    # Use joblib to parallelize the processing
    for file_path in file_text_paths:
        if "3.txt" in file_path:
            print(f"Error file: {file_path}")
            continue
        print(file_path)
        process_file(file_path, output_folder_path)

    # Use joblib to parallelize the processing
    # Parallel(n_jobs=3)(delayed(process_file)(file_path, output_folder_path, name_folder) for file_path in tqdm(file_paths))