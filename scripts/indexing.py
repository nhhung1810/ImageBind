import numpy as np
import os
from typing import List, Tuple
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from datasets import load_dataset, Dataset, load_from_disk
from utils import make_nested_dir, make_random_video_thumbnail
import faiss
import subprocess

METRICS = faiss.METRIC_INNER_PRODUCT
EMB_COLUMN = "video_embeddings"


device = "cpu"
torch.set_grad_enabled(False)
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def calculate_text_embedding(texts: List[str]) -> List[Tuple[str, np.ndarray]]:
    results = []
    text_inp = data.load_and_transform_text(text=texts, device=device)
    embeddings = model({ModalityType.TEXT: text_inp})
    text_embs = embeddings[ModalityType.TEXT].numpy()
    for idx, text in enumerate(texts):
        results.append((text, text_embs[idx]))
        pass

    return results


def calculate_video_embedding(video_path: str) -> torch.Tensor:
    import time

    start = time.perf_counter()
    video_input = data.load_and_transform_video_data(
        video_paths=[video_path],
        device=device,
    )
    # [1, 15, 3, 2, 224, 224]
    # print(video_input.shape)
    inputs = {ModalityType.VISION: video_input}
    with torch.no_grad():
        embeddings = model(inputs)
    # [1, 1024]
    # print(embeddings[ModalityType.VISION].shape)
    end = time.perf_counter()
    print(f"Elapsed time. if {end - start:.3f}s")
    return embeddings[ModalityType.VISION][0].numpy()


def load_video_dataset(save_path, force_reload=True, kwargs={}):
    if force_reload or (not force_reload and not os.path.exists(save_path)):
        return make_video_dataset(save_path, **kwargs)

    return load_from_disk(save_path)


def make_video_dataset(save_path: str, max_rows: int = 10, batch_size=8):
    data = load_dataset("csv", data_files="./snapshot.csv", split=None)
    max_rows = max_rows % data["train"].num_rows
    data = data["train"].select(range(max_rows))
    # Create the path cols
    data: Dataset = data.map(
        lambda x: {
            "path": os.path.join("./datasets/AVIClips", x["clipname"]) + ".avi",
            **x,
        }
    )
    # Processing video embedding
    data = data.map(
        lambda x: {**x, EMB_COLUMN: calculate_video_embedding(x["path"])},
        batch_size=batch_size,
    )
    make_nested_dir(save_path)
    data.save_to_disk(save_path)
    return data


def search_index(data_index: Dataset, text_query: str):
    text_embs = calculate_text_embedding([text_query])
    _, text_emb = text_embs[0]
    # text_emb = torch.rand([1024]).numpy()
    scores, samples = data_index.get_nearest_examples(
        index_name=EMB_COLUMN, query=text_emb, k=5
    )
    paths = samples["path"]
    for score, path in zip(scores, paths):
        print(f"Video {os.path.basename(path)} relavent score: {score}")
        make_random_video_thumbnail(video_path=path)
        pass

    pass


if __name__ == "__main__":
    data_index = load_video_dataset(
        save_path="./_tmp/dataset/video.db", force_reload=False, kwargs={"max_rows": 20}
    )
    data_index.add_faiss_index(EMB_COLUMN, metric_type=METRICS)

    TEXT_QUERY = "Must have both women and men"
    search_index(data_index=data_index, text_query=TEXT_QUERY)
    pass
