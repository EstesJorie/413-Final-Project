import os
from sklearn.model_selection import train_test_split
import random

def europarlToWili(
    europarl_dir,
    output_dir="europarl-v7",
    language_pairs=[("fr", "en")],
    samples_per_lang=5000,
    test_size=0.2,
    shuffle=True,
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    data = []

    for lang1, lang2 in language_pairs:
        for lang in (lang1, lang2):
            file_path = os.path.join(europarl_dir, f"europarl-v7.{lang1}-{lang2}.{lang}")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if shuffle:
                    random.seed(seed)
                    random.shuffle(lines)
                selected_lines = lines[:samples_per_lang]
                labeled = [(line, lang) for line in selected_lines]
                data.extend(labeled)

    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    texts, labels = zip(*data)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed
    )

    def save_list(filepath, lines):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    save_list(os.path.join(output_dir, "x_train.txt"), x_train)
    save_list(os.path.join(output_dir, "y_train.txt"), y_train)
    save_list(os.path.join(output_dir, "x_test.txt"), x_test)
    save_list(os.path.join(output_dir, "y_test.txt"), y_test)

    print(f"Conversion complete. Data saved to: {output_dir}")
    print(f"Languages included: {[lang for pair in language_pairs for lang in pair]}")
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")

# Example usage
if __name__ == "__main__":
    europarlToWili(
        europarl_dir=".",  
        language_pairs=[("fr", "en")],
        samples_per_lang=100000
    )