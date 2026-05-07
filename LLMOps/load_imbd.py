


from pathlib import Path
from datasets import Dataset

def load_imdb_split(path):
    samples = []

    for label_dir in ["pos", "neg"]:
        label = 1 if label_dir == "pos" else 0

        folder = Path(path) / label_dir

        for file in folder.glob("*.txt"):
            text = file.read_text(encoding="utf-8")

            samples.append({
                "text": text,
                "label": label
            })

    return Dataset.from_list(samples)


