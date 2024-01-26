from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# text_list = ["A dog.", "A car", "A bird"]
# image_paths = [
#     ".assets/dog_image.jpg",
#     ".assets/car_image.jpg",
#     ".assets/bird_image.jpg",
# ]
# audio_paths = [
#     ".assets/dog_audio.wav",
#     ".assets/car_audio.wav",
#     ".assets/bird_audio.wav",
# ]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

import time

start = time.perf_counter()
video_input = data.load_and_transform_video_data(
    video_paths=["./datasets/AVIClips/actionclipautoautotrain00001.avi"], device=device
)
# [1, 15, 3, 2, 224, 224]
print(video_input.shape)
inputs = {ModalityType.VISION: video_input}
with torch.no_grad():
    embeddings = model(inputs)
# [1, 1024]
print(embeddings[ModalityType.VISION].shape)
end = time.perf_counter()
print(f"Elapsed time. if {end - start:.3f}s")
