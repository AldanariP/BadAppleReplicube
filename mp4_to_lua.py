import json

import cv2
import numpy as np
from tqdm import trange


def video_to_lua_matrices(input_video_path, target_size=64, threshold=127, output_fps=None):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_skip = 1
    if output_fps is not None and output_fps < original_fps:
        frame_skip = int(original_fps / output_fps)

    frame_count = 0
    frame_groups = np.empty(
        shape=((total_frames // frames_per_group + 1),
            frames_per_group,
            square_resolution,
            square_resolution
        ),
        dtype=np.uint32
    )

    for i in trange(total_frames):
        more, frame = cap.read()
        if not more:
            break

        if frame_count % frame_skip == 0:
            frame_count += 1
            frame_groups[i // 32][i % square_resolution] = (cv2.resize(
                src=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                dsize=(target_size, target_size),
                interpolation=cv2.INTER_AREA
            ) > threshold).astype(np.uint32)

    cap.release()

    powers = 2 ** np.arange(square_resolution - 1, -1, -1)

    with open('template.vox', 'r') as file:
        template = json.load(file)

    for i, group in enumerate(frame_groups):
        code = ("if z == 0 then\n  v_frames = {"
                + ",".join(
                    f"{{{",".join(
                        str(row.dot(powers)) for row in frame
                    )}}}" for frame in
                    group)
                + "}\n  return ((v_frames[t + 1][-y + 16] >> (31 - (x + 16))) & 1) == 1 and BLACK or WHITE\nend")

        template['code'] = code
        template['id'] = str(i)
        template['name'] = f"badapple{str(i)}"

        with open(f"voxs/{i}.vox", 'w') as f:
            f.write(json.dumps(template, indent=0))


if __name__ == "__main__":
    input_video = "original.mp4"
    frames_per_group = 32
    target_fps = 30
    square_resolution = 31
    binary_threshold = 127

    video_to_lua_matrices(input_video, square_resolution, binary_threshold, target_fps)
