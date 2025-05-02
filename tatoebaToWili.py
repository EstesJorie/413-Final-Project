import pandas as pd
import os
from sklearn.model_selection import train_test_split

def processTatoeba(
    input_file="sentences.csv",
    output_dir="tatoeba-sentences",
    languages=["eng", "fra", "deu", "spa"],  # ISO 639-3 codes
    samples_per_lang=5000,
    test_size=0.2,
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(input_file, sep="\t", header=None, names=["id", "lang", "sentence"])
    
    data = []
    for lang in languages:
        subset = df[df["lang"] == lang].dropna()
        if len(subset) < samples_per_lang:
            print(f"⚠️ Not enough samples for language {lang}. Found: {len(subset)}")
            continue
        sampled = subset.sample(n=samples_per_lang, random_state=seed)
        data.extend(zip(sampled["sentence"].tolist(), sampled["lang"].tolist()))

    # Shuffle and split
    print("Splitting into train and test sets.")
    texts, labels = zip(*data)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed
    )

    def save_list(filepath, items):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(items))

    save_list(os.path.join(output_dir, "x_train.txt"), x_train)
    save_list(os.path.join(output_dir, "y_train.txt"), y_train)
    save_list(os.path.join(output_dir, "x_test.txt"), x_test)
    save_list(os.path.join(output_dir, "y_test.txt"), y_test)

    print(f"Done. Saved data to '{output_dir}'")

if __name__ == "__main__":
    processTatoeba(
        input_file="sentences.csv",  #
        languages=["eng", "fra", "deu", "spa", "ita"],
        samples_per_lang=100000
    )