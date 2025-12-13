import os
import logging
import torch
# Import TransNetV2 model class from the same package
from .transnetv2_pytorch import TransNetV2

# Import backend classes for decoding and inference
from . import backend_pyav
from . import backend_nvvc

class SceneDetector:
    """
    SceneDetector is an interface for detecting scene boundaries in videos using the TransNetV2 model.
    It automatically selects between an NVIDIA GPU-accelerated decoding backend (if available) or a CPU-based PyAV backend.
    """
    def __init__(self, device: torch.device | None = None):
        """
        Initialize the SceneDetector.
        If device is not provided, use CUDA if available, otherwise CPU.
        The TransNetV2 model will be loaded on the specified device upon first use.
        """
        # Determine device (CUDA or CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        # Set up a logger for this class
        self.logger = logging.getLogger(__name__)

    def _init_model(self):
        """
        Load the TransNetV2 model and weights onto the specified device.
        Uses torch.compile for optimization if running on CUDA.
        """
        model = TransNetV2()
        # Load model weights from the package directory
        state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transnetv2-pytorch-weights.pth")
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        # Optimize model execution on CUDA
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        return model

    def predict(self, video_path: str, show_progressbar: bool = False):
        """
        Detect scene boundaries in the given video file.

        Parameters:
            video_path (str): Path to the video file to process.
            show_progressbar (bool): If True, display a progress bar during processing.

        Returns:
            scenes (np.ndarray): An array of [start_frame, end_frame] pairs for each detected scene.
        """
        # Initialize the model on first use
        if self.model is None:
            self.model = self._init_model()
            self.logger.info(f"Initialized TransNetV2 model on {self.device.type.upper()} device")

        # If device is CPU or CUDA is not available, use PyAV backend directly
        if self.device.type != "cuda":
            self.logger.debug("Using PyAV backend (CPU decoding)")
            backend = backend_pyav.PyAVBackend()
            scenes = backend.predict_video(video_path, self.model, device=self.device, show_progressbar=show_progressbar)
            return scenes

        # If device is CUDA, attempt to use NVDEC backend for GPU decoding
        try:
            self.logger.debug("Attempting NVDEC backend (GPU decoding)")
            backend = backend_nvvc.NVVCBackend()
            scenes = backend.predict_video(video_path, self.model, device=self.device, show_progressbar=show_progressbar)
            return scenes
        except Exception as e:
            # If any error occurs (e.g., NVDEC not available or decoding fails), fall back to PyAV
            self.logger.warning(f"NVDEC backend failed (error: {e}). Falling back to PyAV backend.")
            backend = backend_pyav.PyAVBackend()
            scenes = backend.predict_video(video_path, self.model, device=self.device, show_progressbar=show_progressbar)
            return scenes