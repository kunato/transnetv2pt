import logging
import av
import numpy as np
import torch
from tqdm import tqdm

class PyAVBackend:
    """
    Backend for video decoding and scene detection using PyAV (FFmpeg).
    This backend decodes video frames on the CPU using PyAV and processes them with the TransNetV2 model.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict_video(self, video_path: str, model: torch.nn.Module, device: torch.device, show_progressbar: bool = False):
        """
        Decode the video using PyAV and use the TransNetV2 model to predict scene cuts.

        Parameters:
            video_path (str): Path to the video file.
            model (torch.nn.Module): Loaded TransNetV2 model for prediction.
            device (torch.device): The device on which to run the model (CPU or CUDA).
            show_progressbar (bool): If True, display a progress bar during frame extraction and processing.

        Returns:
            scenes (np.ndarray): An array of [start_frame, end_frame] pairs for each detected scene.
        """
        # Decode all frames from the video using PyAV
        target_width = 48
        target_height = 27
        frames = self._extract_frames(video_path, target_width, target_height, show_progressbar)
        num_frames = frames.shape[0]
        if num_frames == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Determine padding at start and end (25 frames on each side, plus extra to align to 50)
        pad_start = 25
        pad_end = 25 + 50 - (num_frames % 50 if num_frames % 50 != 0 else 50)
        total_virtual = pad_start + num_frames + pad_end
        num_windows = (total_virtual - 100) // 50 + 1

        # Prepare padded frame sequence (with repeated first and last frame for padding)
        start_frame = frames[0:1]  # first frame
        end_frame = frames[-1:]    # last frame
        padded_frames = np.concatenate([
            np.repeat(start_frame, pad_start, axis=0),
            frames,
            np.repeat(end_frame, pad_end, axis=0)
        ], axis=0)

        # Process frames in windows of 100 with stride 50
        preds_list = []
        frame_windows = range(num_windows)
        if show_progressbar:
            frame_windows = tqdm(frame_windows, total=num_windows, desc="Processing windows", unit="win")
        for i in frame_windows:
            start_idx = i * 50
            batch_frames = padded_frames[start_idx : start_idx + 100]  # shape (100, H, W, 3)
            batch_frames = batch_frames[np.newaxis, ...]               # shape (1, 100, H, W, 3)
            # Move batch to the model's device and run prediction
            batch_tensor = torch.from_numpy(batch_frames).to(device)
            with torch.inference_mode():
                one_hot, _ = model(batch_tensor)
                p = torch.sigmoid(one_hot)[0, 25:75, 0].cpu().numpy()
                preds_list.append(p)

        # Combine predictions from all windows
        single_frame_pred = np.concatenate(preds_list, axis=0)
        # Convert frame-wise predictions to scene boundaries
        scenes = self._predictions_to_scenes(single_frame_pred)
        self.logger.info(f"Detected {len(scenes)} scenes")
        return scenes

    def _extract_frames(self, video_path: str, target_width: int, target_height: int, show_progressbar: bool = False) -> np.ndarray:
        """
        Extract all frames from the video at the specified resolution using PyAV.

        Parameters:
            video_path (str): Path to the video file.
            target_width (int): Width to scale frames to.
            target_height (int): Height to scale frames to.
            show_progressbar (bool): If True, show a progress bar during frame extraction.

        Returns:
            np.ndarray: Array of frames with shape (num_frames, target_height, target_width, 3) in uint8 format.
        """
        self.logger.info(f"Opening video: {video_path}")
        frames_list = []
        try:
            with av.open(video_path) as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream found in file: {video_path}")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                total_frames = stream.frames or None  # total frame count if known
                frame_iterator = container.decode(video=0)
                if show_progressbar:
                    frame_iterator = tqdm(frame_iterator, total=total_frames, desc="Extracting frames", unit="frame")
                for frame in frame_iterator:
                    # Convert frame to RGB and resize to target dimensions
                    frame = frame.reformat(width=target_width, height=target_height, format="rgb24")
                    frame_array = frame.to_ndarray()
                    frames_list.append(frame_array)
        except (av.FFmpegError, OSError, ValueError) as e:
            # Log and re-raise any errors encountered during video reading or decoding
            self.logger.error(f"Failed to open or decode video: {video_path}. PyAV error: {e}")
            raise

        self.logger.info(f"Extracted {len(frames_list)} frames from {video_path}")
        return np.asarray(frames_list, dtype=np.uint8)

    def _predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert an array of frame-level predictions into scene boundary intervals.

        Parameters:
            predictions (np.ndarray): 1D array of scene-cut probabilities for each frame.
            threshold (float): Threshold for detecting a scene cut (default 0.5).

        Returns:
            np.ndarray: Array of [start_frame, end_frame] pairs for each detected scene.
        """
        pred = (predictions > threshold).astype(np.uint8)
        scenes = []
        t_prev = 0
        start = 0
        for i, t in enumerate(pred):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t_prev == 0:
            scenes.append([start, len(pred) - 1])
        if len(scenes) == 0:
            return np.array([[0, len(pred) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)
