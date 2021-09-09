# https://github.com/soCzech/TransNetV2
import torch
from .transnetv2_pytorch import TransNetV2
import ffmpeg
import numpy as np
import os


model = TransNetV2()
state_dict = torch.load(
    f"{os.path.dirname(os.path.abspath(__file__))}/transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval()


def input_iterator(frames):
    # return windows of size 100 where the first/last 25 frames are from the previous/next batch
    # the first and last window must be padded by copies of the first and last frame of the video
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - \
        (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start +
        [frames] + [end_frame] * no_padded_frames_end, 0
    )

    ptr = 0
    while ptr + 100 <= len(padded_inputs):
        out = padded_inputs[ptr:ptr + 100]
        ptr += 50
        yield out[np.newaxis]


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def predict_raw(model, video, device=torch.device('cuda:0')):
    model.to(device)
    with torch.no_grad():
        predictions = []
        for inp in input_iterator(video):
            video_tensor = torch.from_numpy(inp)
            # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
            video_tensor = video_tensor.to(device)

            single_frame_pred, all_frame_pred = model(video_tensor)

            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(
                all_frame_pred["many_hot"]).cpu().numpy()
            predictions.append(
                (single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0]))
            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(video)), len(video)
            ), end="")
        single_frame_pred = np.concatenate(
            [single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate(
            [all_ for single_, all_ in predictions])

        return video.shape[0], single_frame_pred[:len(video)], all_frames_pred[:len(video)]


def predict_video(filename_or_video):
    if isinstance(filename_or_video, str):
        video_stream, err = ffmpeg.input(filename_or_video).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)
        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    else:
        assert filename_or_video.shape[1] == 27 and filename_or_video.shape[2] == 48 and filename_or_video.shape[3] == 3
        video = filename_or_video
    _, single_frame_pred, _ = predict_raw(model, video)
    scenes = predictions_to_scenes(single_frame_pred)
    return scenes
