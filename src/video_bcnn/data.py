"""Contiguous video-clip decoding with temporally consistent face preprocessing."""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor


class UnreadableVideoError(RuntimeError):
    """A video the decoder cannot open at all, after retries."""


def skip_unreadable_collate(batch):
    """Drop unreadable items instead of failing the epoch.

    One corrupt file in 30k training videos is statistically irrelevant, but it
    has now ended three multi-hour runs. Returning None lets the caller count
    and report the skip rather than lose the run.
    """
    items = [item for item in batch if item is not None]
    if not items:
        return None
    return default_collate(items)


def load_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


class VideoClipDataset(Dataset):
    """Decode contiguous clips in memory without writing extracted frames to disk."""

    def __init__(self, records, dataset_roots, config, training, clips_per_video=None):
        self.records = list(records)
        self.dataset_roots = {name: Path(root) for name, root in dataset_roots.items()}
        self.training = bool(training)
        self.clip_length = int(config["clip_length"])
        self.train_clip_strides = tuple(int(value) for value in config.get("train_clip_strides", [1]))
        self.eval_clip_stride = int(config.get("eval_clip_stride", 1))
        default_clips = 1 if self.training else int(config.get("eval_clips_per_video", 1))
        self.clips_per_video = int(default_clips if clips_per_video is None else clips_per_video)
        self.face_crop = bool(config.get("face_crop", True))
        self.face_margin = float(config.get("face_margin", 0.25))
        self.face_detector_scale_factor = float(config.get("face_detector_scale_factor", 1.1))
        self.face_detector_min_neighbors = int(config.get("face_detector_min_neighbors", 5))
        self.box_smoothing_alpha = float(config.get("box_smoothing_alpha", 0.6))
        self.horizontal_flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        # Overfit diagnostics need the same frames every epoch, so the random
        # training clip position can be pinned to the middle of the video.
        self.deterministic_clips = bool(config.get("deterministic_train_clips", False))
        self.resize = Resize(int(config["input_resize"]))
        self.center_crop = CenterCrop(int(config["center_crop"]))
        self.to_tensor = ToTensor()
        self.normalize = Normalize(config["normalize_mean"], config["normalize_std"])
        self.detector = None
        # Repaired frame counts, so a video with bad metadata is measured once
        # per worker rather than on every epoch.
        self.frame_count_cache = {}
        # Videos this worker could not open, reported instead of crashing.
        self.unreadable = set()
        if self.clip_length < 1:
            raise ValueError("clip_length must be positive.")
        if not self.train_clip_strides or min(self.train_clip_strides) < 1:
            raise ValueError("train_clip_strides must contain positive integers.")
        if self.eval_clip_stride < 1 or self.clips_per_video < 1:
            raise ValueError("eval clip stride and clips_per_video must be positive.")
        if not 0.0 <= self.box_smoothing_alpha <= 1.0:
            raise ValueError("box_smoothing_alpha must be in [0, 1].")

    def __len__(self):
        return len(self.records)

    def _detector(self):
        if self.detector is None:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade)
            if self.detector.empty():
                raise RuntimeError("OpenCV Haar face cascade could not be initialized.")
        return self.detector

    @staticmethod
    def _max_start(frame_count, clip_length, stride):
        return max(0, int(frame_count) - 1 - (int(clip_length) - 1) * int(stride))

    def clip_indices(self, frame_count):
        """Return one random training clip or deterministic evaluation clips."""
        if frame_count <= 0:
            raise RuntimeError("Video reports no decodable frames.")
        if self.training and not self.deterministic_clips:
            stride = int(np.random.choice(self.train_clip_strides))
            max_start = self._max_start(frame_count, self.clip_length, stride)
            start = int(np.random.randint(0, max_start + 1)) if max_start else 0
            starts = [start]
        elif self.training:
            stride = int(self.train_clip_strides[0])
            starts = [self._max_start(frame_count, self.clip_length, stride) // 2]
        else:
            stride = self.eval_clip_stride
            max_start = self._max_start(frame_count, self.clip_length, stride)
            starts = np.linspace(0, max_start, self.clips_per_video).round().astype(int).tolist()
        return [
            [min(frame_count - 1, start + offset * stride) for offset in range(self.clip_length)]
            for start in starts
        ]

    @staticmethod
    def _center_square_box(frame):
        height, width = frame.shape[:2]
        side = min(height, width)
        return np.asarray([(width - side) / 2.0, (height - side) / 2.0, side, side], dtype=np.float64)

    def _detect_box(self, frame):
        if not self.face_crop:
            return self._center_square_box(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._detector().detectMultiScale(
            gray,
            scaleFactor=self.face_detector_scale_factor,
            minNeighbors=self.face_detector_min_neighbors,
        )
        if not len(boxes):
            return None
        return np.asarray(max(boxes, key=lambda box: int(box[2]) * int(box[3])), dtype=np.float64)

    @staticmethod
    def _interpolate_missing_boxes(raw_boxes, frames):
        valid = [index for index, box in enumerate(raw_boxes) if box is not None]
        if not valid:
            return [VideoClipDataset._center_square_box(frame) for frame in frames]
        filled = []
        for index, box in enumerate(raw_boxes):
            if box is not None:
                filled.append(box.copy())
                continue
            left = max((item for item in valid if item < index), default=None)
            right = min((item for item in valid if item > index), default=None)
            if left is None:
                filled.append(raw_boxes[right].copy())
            elif right is None:
                filled.append(raw_boxes[left].copy())
            else:
                fraction = float(index - left) / float(right - left)
                filled.append((1.0 - fraction) * raw_boxes[left] + fraction * raw_boxes[right])
        return filled

    def _prepare_boxes(self, frames):
        raw_boxes = [self._detect_box(frame) for frame in frames]
        filled = self._interpolate_missing_boxes(raw_boxes, frames)
        boxes, previous = [], None
        alpha = self.box_smoothing_alpha
        for detected in filled:
            smoothed = detected.copy() if previous is None else alpha * detected + (1.0 - alpha) * previous
            boxes.append(smoothed)
            previous = smoothed
        normalized = []
        for frame, box in zip(frames, boxes):
            frame_height, frame_width = frame.shape[:2]
            x, y, width, height = box
            normalized.append([
                (x + width / 2.0) / float(frame_width),
                (y + height / 2.0) / float(frame_height),
                width / float(frame_width),
                height / float(frame_height),
            ])
        normalized = np.asarray(normalized, dtype=np.float64)
        misses = sum(box is None for box in raw_boxes)
        audit = {
            "any_miss": float(misses > 0),
            "miss_fraction": float(misses) / float(len(raw_boxes)),
            "center_x_jitter": float(normalized[:, 0].std()),
            "center_y_jitter": float(normalized[:, 1].std()),
            "width_jitter": float(normalized[:, 2].std()),
            "height_jitter": float(normalized[:, 3].std()),
        }
        return boxes, audit

    def _crop_box(self, frame, box):
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = [float(value) for value in box]
        x0 = max(0, int(round(x - width * self.face_margin)))
        y0 = max(0, int(round(y - height * self.face_margin)))
        x1 = min(frame_width, int(round(x + width * (1.0 + self.face_margin))))
        y1 = min(frame_height, int(round(y + height * (1.0 + self.face_margin))))
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            return crop
        fallback = self._center_square_box(frame)
        x, y, width, height = [int(round(value)) for value in fallback]
        return frame[y:y + height, x:x + width]

    def _transform(self, bgr_frame, flip):
        image = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        if flip:
            image = ImageOps.mirror(image)
        image = self.resize(image)
        image = self.center_crop(image)
        return self.normalize(self.to_tensor(image))

    @staticmethod
    def _decode_clips(capture, video_path, clip_indices):
        """Seek once per clip, then decode sequentially to preserve frame order.

        Returns None when the seek lands past the last decodable frame, which
        happens whenever container metadata overstates the frame count or the
        installed FFmpeg seeks differently from the one the manifest was built
        with. The caller then falls back to a seek-free pass.
        """
        clips = []
        for indices in clip_indices:
            start, end = int(indices[0]), int(indices[-1])
            wanted = set(int(index) for index in indices)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            decoded, previous = {}, None
            for frame_index in range(start, end + 1):
                ok, frame = capture.read()
                # A stalled read can report success and still hand back None:
                # FFmpeg's interrupt callback fires on timeout after OpenCV has
                # already set the success flag.
                if not ok or frame is None:
                    if previous is None:
                        return None
                    frame = previous.copy()
                if frame_index in wanted:
                    decoded[frame_index] = frame.copy()
                previous = frame
            missing = [index for index in wanted if index not in decoded]
            if missing:
                if previous is None:
                    return None
                for index in missing:
                    decoded[index] = previous.copy()
            clips.append([decoded[int(index)].copy() for index in indices])
        return clips

    @staticmethod
    def _measure_decodable_frames(capture):
        """Count frames the decoder actually yields, ignoring container metadata."""
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while capture.grab():
            count += 1
        return count

    @staticmethod
    def _decode_clips_sequential(capture, video_path, clip_indices):
        """Decode every clip in one forward pass, never seeking mid-video.

        `grab` skips colour conversion for frames no clip wants, so the extra
        cost over seeking is small, and it works on files whose seek index is
        unusable.
        """
        wanted = sorted({int(index) for indices in clip_indices for index in indices})
        if not wanted:
            raise RuntimeError("No frames requested for {}".format(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames, position, target, last = {}, 0, set(wanted), wanted[-1]
        while position <= last:
            if not capture.grab():
                break
            if position in target:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                frames[position] = frame.copy()
            position += 1
        if not frames:
            raise UnreadableVideoError(
                "Video yielded no decodable frames; it is likely truncated or "
                "corrupt: {}".format(video_path)
            )
        available = sorted(frames)
        clips = []
        for indices in clip_indices:
            clip = []
            for index in indices:
                index = int(index)
                if index not in frames:
                    # Clamp onto the nearest frame at or before the request.
                    earlier = [key for key in available if key <= index]
                    index = earlier[-1] if earlier else available[0]
                clip.append(frames[index].copy())
            clips.append(clip)
        return clips

    @staticmethod
    def _open_capture(video_path, attempts=3):
        """Open a video, retrying briefly before giving up.

        A single open can fail transiently under IO contention -- a DFD file
        that failed a full scan opened fine seconds later -- so a retry
        distinguishes a busy filesystem from a genuinely broken file.
        """
        for attempt in range(int(attempts)):
            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                return capture
            capture.release()
            if attempt + 1 < int(attempts):
                time.sleep(0.25 * (attempt + 1))
        raise UnreadableVideoError("Unable to open video: {}".format(video_path))

    def _read_clips(self, video_path):
        capture = self._open_capture(video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        key = str(video_path)
        frame_count = self.frame_count_cache.get(key)
        if frame_count is None:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_indices = self.clip_indices(frame_count)
        try:
            decoded_clips = self._decode_clips(capture, video_path, clip_indices)
            if decoded_clips is None:
                # Metadata or the seek index lied. Measure the real length once,
                # remember it for later epochs, and decode without seeking.
                measured = self._measure_decodable_frames(capture)
                if measured < 1:
                    raise UnreadableVideoError(
                        "Video has no decodable frames; it is likely truncated or "
                        "corrupt: {}".format(video_path)
                    )
                self.frame_count_cache[key] = measured
                clip_indices = self.clip_indices(measured)
                decoded_clips = self._decode_clips_sequential(
                    capture, video_path, clip_indices
                )
        finally:
            capture.release()
        clips, audits = [], []
        for frames in decoded_clips:
            boxes, audit = self._prepare_boxes(frames)
            flip = self.training and np.random.random() < self.horizontal_flip_probability
            tensors = [self._transform(self._crop_box(frame, box), flip) for frame, box in zip(frames, boxes)]
            clips.append(torch.stack(tensors, dim=1))
            audits.append(audit)
        return clips, audits, fps if np.isfinite(fps) and fps > 0.0 else 0.0

    def __getitem__(self, index):
        record = self.records[index]
        video_path = self.dataset_roots[record["dataset"]] / record["path"]
        try:
            clips, audits, fps = self._read_clips(video_path)
        except UnreadableVideoError:
            # Signal the collate function to drop this item. Every other error
            # still propagates, so real bugs are not swallowed.
            self.unreadable.add(str(video_path))
            return None
        result = {
            "label": torch.tensor(int(record["label"]), dtype=torch.long),
            "path": str(video_path),
            "dataset": record["dataset"],
            "method": record["method"],
            "target_id": record.get("target_id", ""),
            "donor_id": record.get("donor_id", ""),
            "fps": torch.tensor(fps, dtype=torch.float32),
            "clip_face_any_miss": torch.tensor([item["any_miss"] for item in audits], dtype=torch.float32),
            "clip_face_miss_fraction": torch.tensor([item["miss_fraction"] for item in audits], dtype=torch.float32),
            "clip_box_center_x_jitter": torch.tensor([item["center_x_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_center_y_jitter": torch.tensor([item["center_y_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_width_jitter": torch.tensor([item["width_jitter"] for item in audits], dtype=torch.float32),
            "clip_box_height_jitter": torch.tensor([item["height_jitter"] for item in audits], dtype=torch.float32),
        }
        if self.training:
            result["clip"] = clips[0]
        else:
            result["clips"] = torch.stack(clips, dim=0)
        return result


class CachedClipDataset(Dataset):
    """Decode each video once and keep the tensors in RAM.

    Only for the overfit diagnostic. Decoding dominates the epoch (Haar
    detection alone is ~93% of per-item time), so caching turns "can this
    architecture fit 40 videos" into a question answered in minutes instead of
    hours. Requires num_workers=0, otherwise each respawned worker starts with
    an empty cache.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.dataset[index]
        return self.cache[index]
