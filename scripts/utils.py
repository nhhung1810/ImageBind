import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from pathlib import Path
import subprocess

IS_TIME_SEED = True
if IS_TIME_SEED:
    import time

    SEED = time.time_ns() % (2**32 - 1)
else:
    SEED = 1810

random_state = np.random.RandomState(seed=SEED)


def make_nested_dir(path: str, is_file=True) -> str:
    if is_file:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
    pass


def make_random_video_thumbnail(video_path: str):
    # THWC
    video_data, audio_data, metadata = torchvision.io.read_video(
        video_path,
        start_pts=0,
        end_pts=60,
        pts_unit="sec",
    )

    # Only get the central
    idx = random_state.randint(
        low=video_data.shape[0] // 4,
        high=video_data.shape[0] - video_data.shape[0] // 4,
    )

    output_name = os.path.basename(video_path).rsplit(".", maxsplit=1)[0]
    output_path = f"./_tmp/thumbnails/{output_name}.png"
    make_nested_dir(output_path)

    # Saving
    save_image_tight(image_data=video_data[idx].numpy(), output_path=output_path)
    print(f"Output thumbnail logged at: {output_path}")

    # Make gif
    gif_name = f"./_tmp/gif/{output_name}.gif"
    make_nested_dir(gif_name)
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            "{input_file}".format(input_file=os.path.abspath(video_path)),
            "-filter:v",
            "setpts=0.1*PTS",
            "{output_file}".format(
                output_file=os.path.abspath(gif_name),
            ),
        ],
        # shell=True,
    )
    pass


def save_image_tight(image_data: np.ndarray, output_path):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(
        image_data,
    )
    fig.savefig(output_path, bbox_inches="tight")
    pass


if __name__ == "__main__":
    make_random_video_thumbnail("./datasets/AVIClips/actionclipautoautotrain00018.avi")
    pass
