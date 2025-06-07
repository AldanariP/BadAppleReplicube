import os

import cv2
import numpy as np
from PIL import Image
from tqdm import trange, tqdm


def merge_gifs(input_folder, output_path, fps):
    print(f"Looking into {input_folder}...")

    files = sorted(int(file[:-4]) for file in os.listdir(input_folder))

    if not files:
        raise FileNotFoundError(f"No GIF files found in folder {input_folder}")

    start = files[0]
    end = files[-1] + 1

    if missing_files := {i for i in range(start, end)} - set(files):
        raise FileNotFoundError(f"Missing GIF files: {', '.join(f'{file}.gif' for file in sorted(missing_files))}")

    print(f"Found {end - start} GIF files")

    frames = []
    for i in trange(start, end, desc="Reading GIFs", unit="files"):
        gif_path = os.path.join(input_folder, f"{i}.gif")
        with Image.open(gif_path) as img:
            for frame_idx in range(getattr(img, 'n_frames')):
                img.seek(frame_idx)
                frame = np.array(img.convert('RGB'))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)

    print(f"Total frames collected: {len(frames)}")

    pil_frames = []
    for frame in tqdm(frames, desc="Converting frames to PIL", unit="frames"):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        pil_frames.append(pil_frame)

    print(f"Saving merged GIF to {output_path}...")

    pil_frames[0].save(
        os.path.join(output_path, f"merged_{fps}fps.gif"),
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000/fps,
        loop=0
    )

    print(f"Successfully merged GIF, total: {len(pil_frames)} frames")


if __name__ == "__main__":
    gif_folder = os.path.join(os.getcwd(), "gifs")
    output_file = os.path.join(os.getcwd())

    merge_gifs(gif_folder, output_file, fps=30)
