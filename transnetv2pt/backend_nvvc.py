import logging
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Try to import PyNvVideoCodec (NVIDIA Video Codec Python bindings)
try:
    import PyNvVideoCodec as nvc
except ImportError:
    nvc = None

class NVVCBackend:
    """
    Backend for video decoding and scene detection using NVIDIA's NVDEC for acceleration.
    This backend decodes video frames on the GPU and processes them with the TransNetV2 model.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if nvc is None:
            # If NVDEC bindings are not available, this backend cannot be used.
            self.logger.error("PyNvVideoCodec (NVDEC) is not available")
            # Raise ImportError so that callers can catch and handle the fallback.
            raise ImportError("PyNvVideoCodec module is not installed or could not be imported")

    def predict_video(self, video_path: str, model: torch.nn.Module, device: torch.device, show_progressbar: bool = False):
        """
        Decode the video using NVDEC and use the TransNetV2 model to predict scene cuts.

        Parameters:
            video_path (str): Path to the video file.
            model (torch.nn.Module): Loaded TransNetV2 model for prediction.
            device (torch.device): The device on which the model is running (must be a CUDA device).
            show_progressbar (bool): If True, display a progress bar for decoding windows.

        Returns:
            scenes (np.ndarray): An array of [start_frame, end_frame] pairs for each detected scene.
        """
        if device.type != "cuda":
            raise RuntimeError("NVVCBackend requires a CUDA device for decoding.")
        # Initialize NVDEC decoder for the input video
        decoder = nvc.SimpleDecoder(enc_file_path=video_path, 
                                    gpu_id=self._get_cuda_gpu_id(device),
                                    use_device_memory=True, 
                                    output_color_type=nvc.OutputColorType.RGB)
        total_frames = len(decoder)
        if total_frames <= 0:
            raise ValueError(f"Empty or invalid video stream: {video_path}")
        # Plan padding for start and end frames
        pad_start = 25
        pad_end = 25 + 50 - (total_frames % 50 if total_frames % 50 != 0 else 50)
        total_virtual = pad_start + total_frames + pad_end
        num_windows = (total_virtual - 100) // 50 + 1

        self.logger.info(f"NVDEC open: {video_path} | frames={total_frames} | windows={num_windows}")

        # Retrieve the first and last frame for padding (as GPU tensors)
        first_frame_gpu = torch.from_dlpack(decoder[0])              # First frame
        last_frame_gpu = torch.from_dlpack(decoder[total_frames - 1])  # Last frame
        # Resize padding frames to the model input size (27x48)
        start_frame_rgb = self._resize_frame(first_frame_gpu, target_h=27, target_w=48)
        end_frame_rgb = self._resize_frame(last_frame_gpu, target_h=27, target_w=48)

        # Prepare to iterate over video frames in windows of 100 (with overlap of 50)
        preds_list: List[np.ndarray] = []
        buffer: List[torch.Tensor] = []
        # Fill the initial buffer with the first 100 frames (including padding at start)
        buffer = self._append_frames(decoder, start_frame_rgb, end_frame_rgb, buffer, 
                                     vi_start=0, count=100, target_h=27, target_w=48, 
                                     pad_start=pad_start, total_frames=total_frames, pad_end=pad_end)
        assert len(buffer) == 100
        next_vi = 100

        # Iterate over each window of 100 frames
        frame_windows = range(num_windows)
        if show_progressbar:
            frame_windows = tqdm(frame_windows, total=num_windows, desc="NVDEC windows", unit="win")
        for _ in frame_windows:
            # Stack buffer list into a batch tensor of shape [1, 100, 27, 48, 3] (uint8)
            batch = torch.stack(buffer, dim=0).unsqueeze(0)
            # Run the model on this batch of frames
            with torch.inference_mode():
                one_hot, _ = model(batch)
                # Apply sigmoid and take the center 50 frame predictions from the 100
                p = torch.sigmoid(one_hot)[0, 25:75, 0].cpu().numpy()
                preds_list.append(p)
            # Slide the window: drop the first 50 frames and decode the next 50
            buffer = buffer[50:]
            buffer = self._append_frames(decoder, start_frame_rgb, end_frame_rgb, buffer, 
                                         vi_start=next_vi, count=50, target_h=27, target_w=48, 
                                         pad_start=pad_start, total_frames=total_frames, pad_end=pad_end)
            next_vi += 50

        # Concatenate all prediction segments into one array
        single_frame_pred = np.concatenate(preds_list, axis=0)
        # Convert frame-wise predictions to scene boundaries
        scenes = self._predictions_to_scenes(single_frame_pred)
        self.logger.info(f"Detected {len(scenes)} scenes")
        return scenes

    def _get_cuda_gpu_id(self, device: torch.device) -> int:
        """Utility to get the CUDA device index (or 0 if unspecified)."""
        if device.type != "cuda":
            return 0
        return 0 if device.index is None else int(device.index)

    def _resize_frame(self, frame_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resize a single frame tensor (uint8 HWC) to the target size using bilinear interpolation.
        The input frame_tensor is expected to be on GPU memory.
        """
        # Convert HWC uint8 [H, W, C] to NCHW float tensor
        x = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        # Resize the image
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        # Clamp values to [0, 255] and convert back to uint8
        x = x.clamp_(0.0, 255.0).to(dtype=torch.uint8)
        # Convert back to HWC layout
        return x.squeeze(0).permute(1, 2, 0).contiguous()

    def _append_frames(self, decoder: "nvc.SimpleDecoder", start_frame_rgb: torch.Tensor, 
                       end_frame_rgb: torch.Tensor, buffer: List[torch.Tensor], 
                       vi_start: int, count: int, target_h: int, target_w: int,
                       pad_start: int, total_frames: int, pad_end: int) -> List[torch.Tensor]:
        """
        Append a range of frames to the buffer list for virtual indices [vi_start, vi_start+count).

        This handles padding frames (using start_frame_rgb or end_frame_rgb) for indices outside the actual video range, 
        and decodes real frames for indices within the video.
        """
        if count == 0:
            return buffer

        # Calculate prefix padding (frames before the real video start)
        prefix = 0
        if vi_start < pad_start:
            prefix = min(pad_start - vi_start, count)
        # Calculate suffix padding (frames beyond the real video end)
        suffix = 0
        end_index = vi_start + count
        actual_end_index = pad_start + total_frames
        if end_index > actual_end_index:
            suffix = min(end_index - actual_end_index, count - prefix)
        # Number of real frames in this range (excluding prefix/suffix)
        real_count = count - prefix - suffix

        # Add prefix padding frames (repeat the first frame)
        for _ in range(prefix):
            buffer.append(start_frame_rgb)
        # Add real video frames by decoding with NVDEC
        if real_count > 0:
            real_start_index = max(0, vi_start - pad_start)
            decoder.seek_to_index(real_start_index)
            batch_frames = decoder.get_batch_frames(batch_size=real_count)
            for frame in batch_frames:
                frame_tensor = torch.from_dlpack(frame)
                frame_resized = self._resize_frame(frame_tensor, target_h=target_h, target_w=target_w)
                buffer.append(frame_resized)
        # Add suffix padding frames (repeat the last frame)
        for _ in range(suffix):
            buffer.append(end_frame_rgb)
        return buffer

    def _predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert an array of frame-level predictions (probabilities) into scene boundary intervals.
        A scene boundary is identified when predictions go from below the threshold to above the threshold.
        
        Returns:
            np.ndarray: Array of [start_frame, end_frame] pairs for each detected scene.
        """
        pred = (predictions > threshold).astype(np.uint8)
        scenes = []
        t_prev = 0
        start = 0
        for i, t in enumerate(pred):
            if t_prev == 1 and t == 0:
                # A new scene starts when we transition from cut (1) to no-cut (0)
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                # Previous scene ends right before this cut
                scenes.append([start, i])
            t_prev = t
        # Handle the last scene after the final cut
        if t_prev == 0:
            scenes.append([start, len(pred) - 1])
        if len(scenes) == 0:
            # If no cuts at all, the entire video is one scene
            return np.array([[0, len(pred) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)
