from langdetect import detect, LangDetectException
import json
import os
from tqdm import tqdm

# Function to load data from the files
def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

datasets = {
    'wili': {
        'train': {'x': "wili-2018/x_train.txt", 'y': "wili-2018/y_train.txt"},
        'test': {'x': "wili-2018/x_test.txt", 'y': "wili-2018/y_test.txt"}
    },
    'europarl': {
        'train': {'x': "europarl-v7/x_train.txt", 'y': "europarl-v7/y_train.txt"},
        'test': {'x': "europarl-v7/x_test.txt", 'y': "europarl-v7/y_test.txt"}
    },
    'tatoeba': {
        'train': {'x': "tatoeba-sentences/x_train.txt", 'y': "tatoeba-sentences/y_train.txt"},
        'test': {'x': "tatoeba-sentences/x_test.txt", 'y': "tatoeba-sentences/y_test.txt"}
    },
    'oscar': {
        'train': {'x': "oscar-data/x_train.txt", 'y': "oscar-data/y_train.txt"},
        'test': {'x': "oscar-data/x_test.txt", 'y': "oscar-data/y_test.txt"}
    }   
}

def is_valid_text(text):
    """Check if text is valid for language detection"""
    return bool(text and text.strip() and any(c.isalpha() for c in text))

def main():
    label_to_language = {}

    try:
        # Process each dataset
        for dataset_name, dataset in datasets.items():
            print(f"\nProcessing {dataset_name} dataset...")
            try:
                x_train = load_data(dataset['train']['x'])
                y_train = load_data(dataset['train']['y'])
            except FileNotFoundError:
                print(f"Warning: Could not find files for {dataset_name}, skipping...")
                continue

            # Process sentences with progress bar
            for idx, (sentence, label) in enumerate(tqdm(zip(x_train, y_train), 
                                                       desc=f"Processing {dataset_name}",
                                                       total=len(x_train))):
                try:
                    # Skip if already mapped or invalid
                    if label in label_to_language or not is_valid_text(sentence):
                        continue
                    
                    # Detect language
                    detected_language = detect(sentence)
                    label_to_language[label] = detected_language
                    
                except LangDetectException:
                    continue
                except Exception as e:
                    print(f"Error processing {dataset_name} sentence {idx}: {str(e)}")
                    continue

        # Save results        
        output_file = os.path.join(f'language_mappings.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(label_to_language, f, indent=4, ensure_ascii=False)
        
        print(f"\nLanguage mappings saved to: {output_file}")
        print(f"Total mappings found: {len(label_to_language)}")
        
        # Print summary
        print("\nLanguage Mapping Summary:")
        for label, lang in sorted(label_to_language.items()):
            print(f"{label}: {lang}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())