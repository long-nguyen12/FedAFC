import json
import os

import pandas as pd
from tqdm import tqdm

DATA_PATH = "C:\\Users\\hnguyen\\Documents\\PhD\\Code\\FedFC\\datasets\\mmcv"


def read_text_corpus(path, subset="all"):
    filename = "mmcv_all.json" if subset == "all" else subset + ".json"
    return pd.read_json(os.path.join(path, filename))


def read_text_evidence_corpus(path):
    text_data = pd.read_json(os.path.join(path, "MMQA_texts.jsonl"), lines=True)
    return text_data[["id", "text"]]


def read_table_evidence_corpus(path):
    """Canonical {title, header, rows} tables; cell links and column metadata are dropped."""
    tables = {}
    with open(os.path.join(path, "MMQA_tables.jsonl"), encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            table = row["table"]
            # Page title and section name say what the table is about; neither is a verdict.
            title = " | ".join(
                part.strip() for part in (row.get("title"), table.get("table_name"))
                if isinstance(part, str) and part.strip()
            )
            tables[row["id"]] = {
                **({"title": title} if title else {}),
                "header": [column["column_name"] for column in table["header"]],
                "rows": [[cell["text"] for cell in cells] for cells in table["table_rows"]],
            }
    return tables


def read_image_path_only(path, mmqa_path):
    image_data = pd.read_json(
        os.path.join(mmqa_path, "MMQA_images.jsonl"), lines=True
    )
    image_data["image"] = image_data["path"].apply(
        lambda filename: os.path.join(path, filename)
    )
    return image_data[["id", "image"]]


def read_images_corpus(path, mmqa_path):
    image_path = os.path.join(path, "final_dataset_images", "final_dataset_images")
    return read_image_path_only(image_path, mmqa_path)


def retrieve_data_for_verification(claims, text_evidence, image_evidence, table_evidence):
    text_data = text_evidence.set_index("id")["text"]
    image_data = image_evidence.set_index("id")["image"]

    claim_data = []
    for claim_id, row in tqdm(claims.iterrows(), total=len(claims)):
        text_evidences = text_data.loc[row["text_evidence"]].values
        image_evidences = image_data.loc[row["image_evidence"]].values
        table_evidences = tuple(table_evidence[table_id] for table_id in row["table_evidence"])

        # MMCV has no claim ID; use the row index in the selected source file.
        claim_object = (
            row["claim"],
            text_evidences,
            image_evidences,
            row["label"],
            claim_id,
            table_evidences,
        )
        claim_data.append(claim_object)

    return claim_data


def get_dataset(path, subset="all", mmqa_path=None):
    """Return every claim as (claim, texts, images, label, claim_id, tables).

    MMCV supplies hop subsets, not train/dev/test splits. Use subset="all"
    or "1hop" through "4hop". The first five fields keep the MOCHEG tuple layout.
    """
    if mmqa_path is None:
        mmqa_path = os.path.join(os.path.dirname(os.path.normpath(path)), "mmqa")

    claims = read_text_corpus(path, subset)
    text_evidence = read_text_evidence_corpus(mmqa_path)
    image_evidence = read_images_corpus(path, mmqa_path)
    table_evidence = read_table_evidence_corpus(mmqa_path)

    return retrieve_data_for_verification(claims, text_evidence, image_evidence, table_evidence)


if __name__ == "__main__":
    claim_data = get_dataset(DATA_PATH)

    print("Dataset size:", len(claim_data))
    print("Claims with table evidence:", sum(bool(claim[5]) for claim in claim_data))
