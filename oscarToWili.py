from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os

if not os.path.exists("oscar_data"):
    os.makedirs("oscar_data")

langs = ["en", "fr", "de", "es"]
data = []

for lang in langs:
    print(f"Loading {lang} dataset...")
    ds = load_dataset("oscar", f"unshuffled_deduplicated_{lang}", split="train", streaming=True)
    count = 0
    for item in ds:
        if count >= 10000:
            break
        item_dict = dict(item)
        text = item_dict["text"].strip().replace("\n", " ")
        if len(text) > 20:  # Only keep sentences with meaningful length
            data.append((text, lang))
            count += 1

print(f"\nTotal samples collected: {len(data)}")

texts, labels = zip(*data)
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels  # Ensure balanced split across languages
)

def save_list(filepath, items):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(items))

print("\nSaving files...")
save_list("oscar_data/x_train.txt", x_train)
save_list("oscar_data/y_train.txt", y_train)
save_list("oscar_data/x_test.txt", x_test)
save_list("oscar_data/y_test.txt", y_test)

print(f"\nFiles saved in oscar_data directory")
