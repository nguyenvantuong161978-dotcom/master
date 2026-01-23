"""
Ken Burns Effect Module - OpenCV Frame-by-Frame Rendering
Chất lượng cao với sub-pixel interpolation (LANCZOS4)

Dựa trên ken_burns_v2.py, tối ưu để tích hợp vào run_edit.py
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Generator, Tuple, List, Optional
from enum import Enum


# ============================================================================
# CONSTANTS
# ============================================================================

class KenBurnsEffect(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN_TOPLEFT = "zoom_in_topleft"
    ZOOM_IN_TOPRIGHT = "zoom_in_topright"
    ZOOM_IN_BOTTOMLEFT = "zoom_in_bottomleft"
    ZOOM_IN_BOTTOMRIGHT = "zoom_in_bottomright"


# Quality presets
QUALITY_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}

TARGET_RATIO = 16 / 9
DEFAULT_FPS = 30


# ============================================================================
# KEN BURNS GENERATOR CLASS
# ============================================================================

# Intensity presets - (zoom_amount, pan_amount, base_scale)
INTENSITY_PRESETS = {
    "minimal": (0.03, 0.15, 1.03),   # Rất nhẹ, giữ gần như toàn bộ ảnh
    "subtle": (0.05, 0.20, 1.05),    # Nhẹ nhàng, êm dịu [MẶC ĐỊNH]
    "light": (0.07, 0.30, 1.07),     # Chuyển động nhẹ
    "normal": (0.10, 0.40, 1.10),    # Chuyển động rõ ràng
    "strong": (0.15, 0.55, 1.15),    # Mạnh, dramatic
}


class KenBurnsCv2:
    """
    Ken Burns Effect Generator using OpenCV
    Renders frame-by-frame with high quality interpolation
    """

    def __init__(
        self,
        output_resolution: str = "1080p",
        fps: int = DEFAULT_FPS,
        fade_duration: float = 0.4,
        intensity: str = "subtle",
        custom_zoom: float = None,
        custom_pan: float = None
    ):
        """
        Args:
            output_resolution: "720p", "1080p", "2k", "4k", or "auto"
            fps: Frames per second
            fade_duration: Fade in/out duration in seconds
            intensity: "minimal", "subtle", "light", "normal", "strong"
            custom_zoom: Custom zoom amount (0.01-0.20), overrides intensity
            custom_pan: Custom pan amount (0.1-0.8), overrides intensity
        """
        self.output_resolution = output_resolution
        self.fps = fps
        self.fade_duration = fade_duration
        self.intensity = intensity.lower()
        self.custom_zoom = custom_zoom
        self.custom_pan = custom_pan
        self._last_effect = None

        # Get intensity settings
        if self.intensity in INTENSITY_PRESETS:
            self.zoom_amount, self.pan_amount, self.base_scale = INTENSITY_PRESETS[self.intensity]
        else:
            self.zoom_amount, self.pan_amount, self.base_scale = INTENSITY_PRESETS["subtle"]

        # Override with custom values if provided
        if custom_zoom is not None:
            self.zoom_amount = max(0.01, min(0.20, custom_zoom))
            self.base_scale = 1.0 + self.zoom_amount
        if custom_pan is not None:
            self.pan_amount = max(0.1, min(0.8, custom_pan))

        # Get output size
        if output_resolution.lower() == "auto":
            self.output_size = None  # Will be set based on input
        else:
            self.output_size = QUALITY_PRESETS.get(
                output_resolution.lower(),
                QUALITY_PRESETS["1080p"]
            )

    def detect_optimal_resolution(self, input_width: int, input_height: int) -> Tuple[int, int]:
        """
        Auto-detect optimal output resolution based on input size

        Rules:
        - Input >= 3840 wide -> 4K
        - Input >= 2560 wide -> 2K
        - Input >= 1920 wide -> 1080p
        - Input >= 1280 wide -> 1080p (slight upscale OK)
        - Input < 1280 -> 720p
        """
        if input_width >= 3840:
            return QUALITY_PRESETS["4k"]
        elif input_width >= 2560:
            return QUALITY_PRESETS["2k"]
        elif input_width >= 1280:
            return QUALITY_PRESETS["1080p"]
        else:
            return QUALITY_PRESETS["720p"]

    def get_effect_params(self, duration: float) -> Tuple[float, float, float]:
        """
        Get zoom/pan amounts based on configured intensity
        Uses instance settings instead of duration-based calculation

        Returns:
            (zoom_amount, pan_amount, base_scale)
        """
        # Use configured values from intensity preset or custom settings
        # Scale slightly based on duration for very short clips
        if duration < 3:
            # Very short: reduce movement slightly
            scale_factor = 0.7
        elif duration < 5:
            scale_factor = 0.85
        else:
            scale_factor = 1.0

        zoom = self.zoom_amount * scale_factor
        pan = self.pan_amount * scale_factor
        base = 1.0 + zoom

        return zoom, pan, base

    def get_random_effect(self) -> KenBurnsEffect:
        """
        Get random effect, avoiding repetition
        Alternates between zoom and pan effects for variety
        """
        effects = list(KenBurnsEffect)

        # Remove last used effect
        if self._last_effect and self._last_effect in effects:
            effects.remove(self._last_effect)

        # Alternate between zoom and pan
        zoom_effects = [e for e in effects if 'zoom' in e.value.lower()]
        pan_effects = [e for e in effects if 'pan' in e.value.lower()]

        if self._last_effect:
            if 'zoom' in self._last_effect.value.lower():
                # Last was zoom, prefer pan
                candidates = pan_effects if pan_effects else effects
            else:
                # Last was pan, prefer zoom
                candidates = zoom_effects if zoom_effects else effects
        else:
            candidates = effects

        effect = random.choice(candidates)
        self._last_effect = effect
        return effect

    def crop_to_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to 16:9 aspect ratio, keeping center
        """
        h, w = image.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - TARGET_RATIO) < 0.01:
            return image

        if current_ratio > TARGET_RATIO:
            # Too wide - crop sides
            new_w = int(h * TARGET_RATIO)
            offset = (w - new_w) // 2
            return image[:, offset:offset + new_w]
        else:
            # Too tall - crop top/bottom
            new_h = int(w / TARGET_RATIO)
            offset = (h - new_h) // 2
            return image[offset:offset + new_h, :]

    def prepare_image(self, image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Prepare image for Ken Burns effect:
        1. Crop to 16:9
        2. Scale up if needed (for zoom/pan headroom)

        Uses configured zoom/pan amounts to determine minimum scale needed
        """
        # Crop to 16:9
        img = self.crop_to_aspect_ratio(image)

        h, w = img.shape[:2]
        out_w, out_h = output_size

        # Calculate minimum scale needed based on intensity settings
        # Only need enough headroom for the configured zoom/pan amount
        # Add small buffer (1.02x) for safety
        headroom = 1.0 + self.zoom_amount + 0.02

        min_scale = max(
            (out_w * headroom) / w,
            (out_h * headroom) / h
        )

        if min_scale > 1:
            new_w = int(w * min_scale)
            new_h = int(h * min_scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return img

    def apply_transform(
        self,
        image: np.ndarray,
        scale: float,
        offset_x: float,
        offset_y: float,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply zoom/pan transform with sub-pixel precision

        Args:
            image: Prepared source image
            scale: Zoom level (1.0 = fit, >1 = zoom in)
            offset_x, offset_y: Pan position (0-1, 0.5 = center)
            output_size: (width, height) of output frame

        Returns:
            Transformed frame
        """
        h, w = image.shape[:2]
        out_w, out_h = output_size

        # Calculate crop region
        crop_w = out_w / scale
        crop_h = out_h / scale

        # Calculate center position
        max_offset_x = max(0, w - crop_w)
        max_offset_y = max(0, h - crop_h)

        center_x = offset_x * max_offset_x + crop_w / 2
        center_y = offset_y * max_offset_y + crop_h / 2

        # Calculate corners
        x1 = center_x - crop_w / 2
        y1 = center_y - crop_h / 2
        x2 = center_x + crop_w / 2
        y2 = center_y + crop_h / 2

        # Source points (float for sub-pixel accuracy)
        src_pts = np.float32([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ])

        # Destination points
        dst_pts = np.float32([
            [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
        ])

        # Perspective transform with high quality interpolation
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(
            image, matrix, (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT
        )

        return result

    def apply_fade(
        self,
        frame: np.ndarray,
        frame_num: int,
        total_frames: int,
        fade_frames: int
    ) -> np.ndarray:
        """
        Apply fade in/out to frame
        """
        if fade_frames <= 0:
            return frame

        alpha = 1.0

        if frame_num < fade_frames:
            # Fade in
            alpha = frame_num / fade_frames
        elif frame_num >= total_frames - fade_frames:
            # Fade out
            alpha = (total_frames - frame_num - 1) / fade_frames

        if alpha < 1.0:
            alpha = max(0.0, min(1.0, alpha))
            # Fade to black
            frame = (frame * alpha).astype(np.uint8)

        return frame

    def generate_frames(
        self,
        image: np.ndarray,
        effect: KenBurnsEffect,
        duration: float,
        output_size: Tuple[int, int],
        apply_fade: bool = True
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate all frames for one image with Ken Burns effect

        Args:
            image: Source image (BGR)
            effect: Ken Burns effect to apply
            duration: Duration in seconds
            output_size: (width, height) of output
            apply_fade: Whether to apply fade in/out

        Yields:
            Frames as numpy arrays (BGR)
        """
        total_frames = int(duration * self.fps)
        fade_frames = int(self.fade_duration * self.fps) if apply_fade else 0

        # Prepare image
        img = self.prepare_image(image, output_size)

        # Get effect parameters
        zoom_amount, pan_amount, base_scale = self.get_effect_params(duration)

        for frame_num in range(total_frames):
            # Progress: 0 -> 1
            t = frame_num / (total_frames - 1) if total_frames > 1 else 0

            # Calculate transform based on effect
            scale, offset_x, offset_y = self._calculate_transform(
                effect, t, zoom_amount, pan_amount, base_scale
            )

            # Apply transform
            frame = self.apply_transform(img, scale, offset_x, offset_y, output_size)

            # Apply fade
            if apply_fade:
                frame = self.apply_fade(frame, frame_num, total_frames, fade_frames)

            yield frame

    def _calculate_transform(
        self,
        effect: KenBurnsEffect,
        t: float,
        zoom_amount: float,
        pan_amount: float,
        base_scale: float
    ) -> Tuple[float, float, float]:
        """
        Calculate scale and offset for given effect and time

        Returns:
            (scale, offset_x, offset_y)
        """
        if effect == KenBurnsEffect.ZOOM_IN:
            scale = 1.0 + zoom_amount * t
            offset_x = 0.5
            offset_y = 0.5

        elif effect == KenBurnsEffect.ZOOM_OUT:
            scale = (1.0 + zoom_amount) - zoom_amount * t
            offset_x = 0.5
            offset_y = 0.5

        elif effect == KenBurnsEffect.PAN_LEFT:
            scale = base_scale
            offset_x = 0.5 + pan_amount/2 - pan_amount * t
            offset_y = 0.5

        elif effect == KenBurnsEffect.PAN_RIGHT:
            scale = base_scale
            offset_x = 0.5 - pan_amount/2 + pan_amount * t
            offset_y = 0.5

        elif effect == KenBurnsEffect.PAN_UP:
            scale = base_scale
            offset_x = 0.5
            offset_y = 0.5 + pan_amount/2 - pan_amount * t

        elif effect == KenBurnsEffect.PAN_DOWN:
            scale = base_scale
            offset_x = 0.5
            offset_y = 0.5 - pan_amount/2 + pan_amount * t

        elif effect == KenBurnsEffect.ZOOM_IN_TOPLEFT:
            scale = 1.0 + zoom_amount * t
            offset_x = 0.3 - 0.15 * t
            offset_y = 0.3 - 0.15 * t

        elif effect == KenBurnsEffect.ZOOM_IN_TOPRIGHT:
            scale = 1.0 + zoom_amount * t
            offset_x = 0.7 + 0.15 * t
            offset_y = 0.3 - 0.15 * t

        elif effect == KenBurnsEffect.ZOOM_IN_BOTTOMLEFT:
            scale = 1.0 + zoom_amount * t
            offset_x = 0.3 - 0.15 * t
            offset_y = 0.7 + 0.15 * t

        elif effect == KenBurnsEffect.ZOOM_IN_BOTTOMRIGHT:
            scale = 1.0 + zoom_amount * t
            offset_x = 0.7 + 0.15 * t
            offset_y = 0.7 + 0.15 * t

        else:
            # Default: zoom_in
            scale = 1.0 + zoom_amount * t
            offset_x = 0.5
            offset_y = 0.5

        return scale, offset_x, offset_y

    def create_clip_from_image(
        self,
        image_path: Path,
        output_path: Path,
        duration: float,
        effect: Optional[KenBurnsEffect] = None,
        output_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Create a video clip from a single image with Ken Burns effect

        Args:
            image_path: Path to source image
            output_path: Path for output video
            duration: Duration in seconds
            effect: Ken Burns effect (None = random)
            output_size: Output size (None = use preset)

        Returns:
            True if successful
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"    Cannot read image: {image_path}")
            return False

        # Determine output size
        if output_size is None:
            if self.output_size is None:
                # Auto mode
                h, w = img.shape[:2]
                output_size = self.detect_optimal_resolution(w, h)
            else:
                output_size = self.output_size

        # Get effect
        if effect is None:
            effect = self.get_random_effect()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, output_size
        )

        if not writer.isOpened():
            print(f"    Cannot create video writer: {output_path}")
            return False

        try:
            # Generate and write frames
            for frame in self.generate_frames(img, effect, duration, output_size):
                writer.write(frame)

            return True
        finally:
            writer.release()

    def process_video_clip(
        self,
        video_path: Path,
        output_path: Path,
        target_duration: float,
        output_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Process video clip: scale to output size with high quality
        No Ken Burns effect for videos, just high quality scaling

        Args:
            video_path: Source video path
            output_path: Output video path
            target_duration: Target duration (will trim if longer)
            output_size: Output size (None = use preset)

        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    Cannot open video: {video_path}")
            return False

        try:
            # Get video properties
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_duration = src_frames / src_fps if src_fps > 0 else 0

            # Determine output size
            if output_size is None:
                if self.output_size is None:
                    output_size = self.detect_optimal_resolution(src_width, src_height)
                else:
                    output_size = self.output_size

            out_w, out_h = output_size

            # Calculate trimming if needed
            if src_duration > target_duration:
                trim_total = src_duration - target_duration
                start_frame = int((trim_total / 2) * src_fps)
                end_frame = start_frame + int(target_duration * src_fps)
            else:
                start_frame = 0
                end_frame = src_frames

            # Calculate fade frames
            fade_frames = int(self.fade_duration * self.fps)
            total_output_frames = int(target_duration * self.fps)

            # Create writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, output_size)

            if not writer.isOpened():
                return False

            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_num = 0
            output_frame_num = 0

            while cap.isOpened() and frame_num < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resample to output fps
                src_time = frame_num / src_fps
                out_time = output_frame_num / self.fps

                if src_time >= out_time:
                    # Scale frame with high quality
                    if (src_width, src_height) != output_size:
                        frame = self._scale_frame_high_quality(frame, output_size)

                    # Apply fade
                    frame = self.apply_fade(frame, output_frame_num, total_output_frames, fade_frames)

                    writer.write(frame)
                    output_frame_num += 1

                frame_num += 1

                if output_frame_num >= total_output_frames:
                    break

            return True

        finally:
            cap.release()
            if 'writer' in locals():
                writer.release()

    def _scale_frame_high_quality(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale frame with high quality, maintaining aspect ratio
        """
        h, w = frame.shape[:2]
        out_w, out_h = output_size

        # Calculate scale to fit
        scale = min(out_w / w, out_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with LANCZOS4
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create output frame with padding (letterbox/pillarbox)
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Center the resized frame
        x_offset = (out_w - new_w) // 2
        y_offset = (out_h - new_h) // 2

        output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return output


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_quality_preset(name: str) -> Tuple[int, int]:
    """Get output size from preset name"""
    return QUALITY_PRESETS.get(name.lower(), QUALITY_PRESETS["1080p"])


def list_effects() -> List[str]:
    """List all available effects"""
    return [e.value for e in KenBurnsEffect]


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Ken Burns CV2 Module")
    print("=" * 40)
    print(f"Available effects: {list_effects()}")
    print(f"Quality presets: {list(QUALITY_PRESETS.keys())}")

    if len(sys.argv) >= 3:
        input_image = sys.argv[1]
        output_video = sys.argv[2]
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

        print(f"\nProcessing: {input_image}")
        print(f"Output: {output_video}")
        print(f"Duration: {duration}s")

        kb = KenBurnsCv2(output_resolution="1080p")
        success = kb.create_clip_from_image(
            Path(input_image),
            Path(output_video),
            duration
        )

        print(f"Result: {'Success' if success else 'Failed'}")
