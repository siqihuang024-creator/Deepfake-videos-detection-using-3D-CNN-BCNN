#!/usr/bin/env python
"""Measure whether a face detector can carry the crop-based pipeline.

Switching from whole frames to face crops moves the failure mode: a whole frame
is at least a real frame, whereas a missed or wandering box feeds the network a
wrong crop and raises no error. So the detector has to be measured before its
boxes are cached, not after a training run comes back at chance.

Three sampling modes, because they answer different questions and the published
protocol is not what our loader does.

  protocol   32 frames spread evenly over the whole video, which is step 1 of
             the DeepfakeBench pipeline the Celeb-DF++ baselines were produced
             under. Frames are scored independently there, so one miss costs one
             frame.

  train      Contiguous clips drawn the way VideoClipDataset.clip_indices draws
             them: clip_length frames at a stride from train_clip_strides, start
             uniform over the whole video. A 3D CNN needs temporal continuity,
             so we cannot use 32 independent frames, and that changes what a
             miss costs -- consecutive frames fail together, and a clip is only
             as good as its worst frame.

  eval       The deterministic clips scoring uses: clips_per_video starts spread
             over the video at eval_clip_stride.

The clip modes therefore report clip-level outcomes as well as frame rates. A
clip missing every box is unusable; a clip missing a few is repaired by the
loader's interpolation and EMA smoothing.
"""

import argparse
import collections
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


# --------------------------------------------------------------------------
# detectors: detect() returns [(x, y, w, h), ...] in pixels; detect_batch()
# exists so a GPU detector can take a whole clip at once.
# --------------------------------------------------------------------------

class BaseDetector(object):
    def detect_batch(self, frames):
        return [self.detect(frame) for frame in frames]


class HaarDetector(BaseDetector):
    name = "haar"
    note = "cv2 cascade, the loader's current default"

    def __init__(self, args):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor = args.haar_scale_factor
        self.min_neighbors = args.haar_min_neighbors

    @staticmethod
    def available(args):
        del args
        return True, ""

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)
        return [tuple(int(v) for v in box) for box in found]


class YuNetDetector(BaseDetector):
    name = "yunet"
    note = "OpenCV CNN detector, no extra dependency"

    def __init__(self, args):
        # The model is size-bound at construction, so it is rebuilt whenever a
        # frame of a new shape arrives rather than per frame.
        self.model_path = str(args.yunet_model)
        self.threshold = args.yunet_threshold
        self.detector = None
        self.shape = None

    @staticmethod
    def available(args):
        if not hasattr(cv2, "FaceDetectorYN"):
            return False, "OpenCV {} has no FaceDetectorYN".format(cv2.__version__)
        if not Path(args.yunet_model).exists():
            return False, "model not found at {}; download it from {}".format(
                args.yunet_model, YUNET_URL)
        # The ONNX has to match the bundled DNN backend: a model newer than the
        # installed OpenCV loads and then throws on the first detect, which
        # would otherwise surface hours into a run. Prove it works here.
        try:
            probe = cv2.FaceDetectorYN_create(
                str(args.yunet_model), "", (64, 64), args.yunet_threshold, 0.3, 10)
            probe.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        except cv2.error as error:
            return False, "model incompatible with OpenCV {} ({})".format(
                cv2.__version__, str(error).strip().splitlines()[-1][:60])
        return True, ""

    def detect(self, frame):
        height, width = frame.shape[:2]
        if self.shape != (width, height):
            self.detector = cv2.FaceDetectorYN_create(
                self.model_path, "", (width, height), self.threshold, 0.3, 5000)
            self.shape = (width, height)
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


class DlibDetector(BaseDetector):
    """The protocol detector: dlib's HOG frontal detector.

    DeepfakeBench performs "face detection, face cropping, and alignment ...
    using DLIB" (paper, Sec. 4.1) with the 81-point shape predictor its README
    asks you to download. This class only needs the box, since detection rate
    and box stability are what the comparison measures; the landmarks matter at
    caching time, where alignment consumes them.
    """

    name = "dlib"
    note = "DeepfakeBench's own detector (paper Sec. 4.1)"

    def __init__(self, args):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.upsample = args.dlib_upsample

    @staticmethod
    def available(args):
        del args
        try:
            import dlib  # noqa: F401
        except Exception as error:
            return False, "needs dlib: conda install -c conda-forge dlib ({})".format(error)
        return True, ""

    def detect(self, frame):
        # dlib wants RGB and, unlike the CNN detectors, gains a lot from an
        # upsampling pass when the face is small relative to the frame.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return [(r.left(), r.top(), r.width(), r.height())
                for r in self.detector(rgb, self.upsample)]


class MtcnnDetector(BaseDetector):
    name = "mtcnn"
    note = "facenet-pytorch, GPU, batched per clip"

    def __init__(self, args):
        import torch
        from facenet_pytorch import MTCNN
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MTCNN(keep_all=True, device=self.device)

    @staticmethod
    def available(args):
        del args
        try:
            import facenet_pytorch  # noqa: F401
        except Exception as error:
            return False, "pip install facenet-pytorch ({})".format(error)
        return True, ""

    @staticmethod
    def _convert(boxes):
        if boxes is None:
            return []
        return [(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                for x0, y0, x1, y1 in boxes]

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.model.detect(rgb)
        return self._convert(boxes)

    def detect_batch(self, frames):
        # Frames within one clip share a shape, which is what lets MTCNN take
        # them as a batch; a mixed batch would raise, so fall back on shape
        # disagreement rather than assume.
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            return [self.detect(frame) for frame in frames]
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        with self.torch.no_grad():
            batch, _ = self.model.detect(rgb)
        if batch is None:
            return [[] for _ in frames]
        return [self._convert(boxes) for boxes in batch]


class RetinaFaceDetector(BaseDetector):
    name = "retinaface"
    note = "facexlib ResNet50, the detector Yermakov et al. 2025 use"

    def __init__(self, args):
        import torch
        from facexlib.detection import init_detection_model
        self.torch = torch
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = init_detection_model("retinaface_resnet50", half=False,
                                          device=self.device)
        self.threshold = args.retinaface_threshold

    @staticmethod
    def available(args):
        del args
        try:
            import facexlib  # noqa: F401
        except Exception as error:
            return False, "pip install facexlib ({})".format(error)
        return True, ""

    def detect(self, frame):
        # facexlib 0.3.0 calls .numpy() on a tensor that still tracks grad, so
        # no_grad here is load-bearing rather than an optimisation. The model
        # subtracts a BGR mean, so the frame goes in as OpenCV produced it.
        with self.torch.no_grad():
            found = self.model.detect_faces(frame, self.threshold)
        rows = np.asarray(found, dtype=np.float32).reshape(-1, 15)
        return [(int(r[0]), int(r[1]), int(r[2] - r[0]), int(r[3] - r[1])) for r in rows]


DETECTORS = [HaarDetector, YuNetDetector, DlibDetector, MtcnnDetector, RetinaFaceDetector]


# --------------------------------------------------------------------------
# clip sampling: mirrors VideoClipDataset.clip_indices in src/video_bcnn/data.py
# --------------------------------------------------------------------------

def max_start(frame_count, clip_length, stride):
    return max(0, int(frame_count) - 1 - (int(clip_length) - 1) * int(stride))


def clip_index_sets(mode, frame_count, args, rng):
    """Frame indices per clip for one sampling mode, spanning the whole video."""
    if mode == "protocol":
        # Step 1 of the published pipeline: evenly spread, frames independent.
        return [np.linspace(0, frame_count - 1, args.protocol_frames).round()
                .astype(int).tolist()]
    length = args.clip_length
    if mode == "train":
        sets = []
        for _ in range(args.train_clips):
            stride = int(rng.choice(args.train_clip_strides))
            top = max_start(frame_count, length, stride)
            start = int(rng.randint(0, top)) if top else 0
            sets.append([min(frame_count - 1, start + i * stride) for i in range(length)])
        return sets
    stride = args.eval_clip_stride
    top = max_start(frame_count, length, stride)
    starts = np.linspace(0, top, args.eval_clips).round().astype(int).tolist()
    return [[min(frame_count - 1, s + i * stride) for i in range(length)] for s in starts]


def read_clips(video_path, index_sets):
    """Decode each clip with one seek, then sequential reads."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    clips = []
    for indices in index_sets:
        frames, wanted = [], list(indices)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(wanted[0]))
        position = int(wanted[0])
        for target in wanted:
            while position < target:
                if not capture.grab():
                    break
                position += 1
            ok, frame = capture.read()
            position += 1
            if ok:
                frames.append(frame)
        if frames:
            clips.append(frames)
    capture.release()
    return clips


def load_sizes(path):
    sizes = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                continue
            key = normalise(row["relative_path"])
            if row["dataset"] == "CelebDFv3":
                sizes[("CelebDFv3", key[len("CelebDFv3/"):])] = (
                    int(row["width"]), int(row["height"]))
            else:
                sizes[("DFD", key)] = (int(row["width"]), int(row["height"]))
    return sizes


def normalise(path):
    return path.replace("\\", "/").lstrip("/")


def shape_group(width, height):
    return "wide" if width / float(height) >= 1.65 else "square"


def build_groups(manifest, sizes, split, per_group, seed):
    """Sample videos per (corpus, class, shape); only the given split is read."""
    buckets = collections.defaultdict(list)
    with open(manifest, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            key = (row["dataset"], normalise(row["path"]))
            if key not in sizes:
                continue
            width, height = sizes[key]
            buckets[(row["dataset"], row["class_name"],
                     shape_group(width, height))].append(row)
    groups, rng = {}, random.Random(seed)
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r["path"])
        rng.shuffle(rows)
        groups[key] = rows[:per_group]
    return groups


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def score_clips(detector, clips):
    """Frame- and clip-level outcomes for one video under one detector.

    Jitter is measured within a clip, never across the video: box movement over
    ten seconds is the subject moving, whereas movement inside eight consecutive
    frames is the detector disagreeing with itself, and only the second is a
    defect the EMA smoother has to absorb.
    """
    frames_total = frames_found = clips_complete = clips_empty = 0
    pixels, jitters, elapsed = [], [], 0.0
    for frames in clips:
        start = time.time()
        per_frame = detector.detect_batch(frames)
        elapsed += time.time() - start
        centres, widths = [], []
        for frame, boxes in zip(frames, per_frame):
            frames_total += 1
            if not boxes:
                continue
            height, width = frame.shape[:2]
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            frames_found += 1
            centres.append(((x + w / 2.0) / width, (y + h / 2.0) / height))
            widths.append(float(w))
        if not widths:
            clips_empty += 1
            continue
        if len(widths) == len(frames):
            clips_complete += 1
        pixels.append(float(np.median(widths)))
        if len(centres) > 1:
            centres = np.asarray(centres)
            jitters.append(float(np.hypot(centres[:, 0].std(), centres[:, 1].std())))
    record = {
        "frames": frames_total, "detected": frames_found,
        "clips": len(clips), "clips_complete": clips_complete,
        "clips_empty": clips_empty,
        "seconds_per_frame": elapsed / max(frames_total, 1),
    }
    if pixels:
        record["box_width_px"] = float(np.median(pixels))
        # Absolute pixels, not the frame fraction: resizing every box to the
        # network input normalises away how much of the frame the face filled,
        # but not the factor each class was resampled by.
        record["scale_to_input"] = None
        record["clip_jitter"] = float(np.median(jitters)) if jitters else float("nan")
    return record


def summarise(rows, target):
    fields = ["detector", "mode", "dataset", "class_name", "shape"]
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key in sorted(buckets):
        group = buckets[key]
        frames = sum(r["frames"] for r in group)
        clips = sum(r["clips"] for r in group)
        scored = [r for r in group if "box_width_px" in r]
        box = float(np.median([r["box_width_px"] for r in scored])) if scored else float("nan")
        entry = dict(zip(fields, key))
        entry.update({
            "videos": len(group),
            "frame_detection_rate": sum(r["detected"] for r in group) / float(max(frames, 1)),
            "clips_complete_rate": sum(r["clips_complete"] for r in group) / float(max(clips, 1)),
            "clips_empty_rate": sum(r["clips_empty"] for r in group) / float(max(clips, 1)),
            "median_box_width_px": box,
            "median_scale_to_input": target / max(1.3 * box, 1e-9) if scored else float("nan"),
            "median_clip_jitter": float(np.nanmedian(
                [r["clip_jitter"] for r in scored])) if scored else float("nan"),
            "seconds_per_frame": float(np.mean([r["seconds_per_frame"] for r in group])),
        })
        out.append(entry)
    return out


def print_table(summary, target):
    head = ("{:<7} {:<8} {:<10} {:<5} {:<6} {:>5} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7}".format(
        "det", "mode", "corpus", "class", "shape", "vids", "frame%",
        "clip ok%", "clip 0%", "box px", "->" + str(target), "jitter"))
    print(head)
    print("-" * len(head))
    for row in summary:
        print("{:<7} {:<8} {:<10} {:<5} {:<6} {:>5} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7}".format(
            row["detector"], row["mode"], row["dataset"], row["class_name"],
            row["shape"], row["videos"],
            "{:.1f}".format(100 * row["frame_detection_rate"]),
            "{:.1f}".format(100 * row["clips_complete_rate"]),
            "{:.1f}".format(100 * row["clips_empty_rate"]),
            "{:.0f}".format(row["median_box_width_px"]),
            "{:.2f}".format(row["median_scale_to_input"]),
            "{:.3f}".format(row["median_clip_jitter"])))


def report_class_gap(summary):
    """Does the resampling factor separate fakes from reals?

    Resizing every box to the network input normalises away how much of the
    frame the face filled, so a class difference in box fraction is harmless.
    What resizing cannot remove is the factor each class was resampled by: if
    fakes arrive consistently upsampled and reals downsampled, the blur and
    ringing that leaves behind is a label cue of the kind the padding share was.
    """
    print("\nResampling gap between classes (crop-scale leak check)")
    index = {(r["detector"], r["mode"], r["dataset"], r["shape"], r["class_name"]): r
             for r in summary}
    for detector, mode, dataset, shape in sorted(
            {(r["detector"], r["mode"], r["dataset"], r["shape"]) for r in summary}):
        real = index.get((detector, mode, dataset, shape, "real"))
        fake = index.get((detector, mode, dataset, shape, "fake"))
        if not real or not fake:
            continue
        a, b = real["median_scale_to_input"], fake["median_scale_to_input"]
        if np.isnan(a) or np.isnan(b):
            continue
        ratio = b / a if a else float("nan")
        flag = "" if 0.8 <= ratio <= 1.25 else "   <-- check"
        print("  {:<7} {:<8} {:<10} {:<6} real x{:.2f}  fake x{:.2f}  ratio {:.2f}{}".format(
            detector, mode, dataset, shape, a, b, ratio, flag))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",
                        default=str(ROOT / "artifacts/manifests/combined_manifest_supervised.csv"))
    parser.add_argument("--sizes",
                        default=str(ROOT / "scripts/video_size_reports/video_sizes.csv"))
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--videos-per-group", type=int, default=30)
    parser.add_argument("--modes", nargs="+", default=["protocol", "train", "eval"],
                        choices=["protocol", "train", "eval"])
    parser.add_argument("--detectors", nargs="+", default=["mtcnn", "retinaface"],
                        help="Names to run; pass 'haar mtcnn' to keep the incumbent "
                             "in the table.")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Network input the crop is resized to. The published "
                             "pipeline fixes the 1.3 margin but not this, which "
                             "follows the backbone: 224 in recent work, 256 as "
                             "DeepfakeBench stores them, 299 for Xception.")
    parser.add_argument("--protocol-frames", type=int, default=32)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--train-clips", type=int, default=4)
    parser.add_argument("--train-clip-strides", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--eval-clips", type=int, default=4)
    parser.add_argument("--eval-clip-stride", type=int, default=2)
    parser.add_argument("--output-dir", default=str(ROOT / "scripts/face_detector_reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--haar-scale-factor", type=float, default=1.1)
    parser.add_argument("--haar-min-neighbors", type=int, default=5)
    parser.add_argument("--yunet-model",
                        default=str(ROOT / "scripts/models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--yunet-threshold", type=float, default=0.6)
    parser.add_argument("--dlib-upsample", type=int, default=1)
    parser.add_argument("--retinaface-threshold", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    roots = {}
    for entry in (args.dataset_root or []):
        name, _, path = entry.partition("=")
        roots[name] = Path(path)
    if not roots:
        import yaml
        with open(ROOT / "configs/combined_3d_bcnn.yaml", "r", encoding="utf-8") as handle:
            roots = {k: Path(v) for k, v in
                     yaml.safe_load(handle)["data"]["dataset_roots"].items()}
    for name, path in roots.items():
        if not path.exists():
            raise SystemExit("dataset root {} does not exist: {}".format(name, path))

    active = []
    for cls in DETECTORS:
        if cls.name not in args.detectors:
            continue
        ok, why = cls.available(args)
        if ok:
            active.append(cls(args))
            print("using   {:<9} {}".format(cls.name, cls.note))
        else:
            print("skipped {:<9} {}".format(cls.name, why))
    if not active:
        raise SystemExit("No detector available out of {}.".format(args.detectors))

    sizes = load_sizes(args.sizes)
    groups = build_groups(args.manifest, sizes, args.split, args.videos_per_group, args.seed)
    total = sum(len(v) for v in groups.values())
    print("\n{} videos, {} groups, modes {}, {} detector(s), target {}px\n".format(
        total, len(groups), ",".join(args.modes), len(active), args.target_size))

    rows, done = [], 0
    rng = np.random.RandomState(args.seed)
    for (dataset, class_name, shape), records in sorted(groups.items()):
        for record in records:
            path = roots[dataset] / record["path"]
            capture = cv2.VideoCapture(str(path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            done += 1
            if frame_count <= 0:
                print("  unreadable: {}".format(record["path"]))
                continue
            for mode in args.modes:
                clips = read_clips(path, clip_index_sets(mode, frame_count, args, rng))
                if not clips:
                    continue
                for detector in active:
                    entry = score_clips(detector, clips)
                    entry.update({"detector": detector.name, "mode": mode,
                                  "dataset": dataset, "class_name": class_name,
                                  "shape": shape, "path": record["path"],
                                  "method": record["method"]})
                    rows.append(entry)
            if done % 20 == 0:
                print("  {}/{} videos".format(done, total))

    summary = summarise(rows, args.target_size)
    print("")
    print_table(summary, args.target_size)
    report_class_gap(summary)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["detector", "mode", "dataset", "class_name", "shape", "method", "path",
              "frames", "detected", "clips", "clips_complete", "clips_empty",
              "box_width_px", "clip_jitter", "seconds_per_frame"]
    with open(output / "per_video.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"settings": vars(args), "groups": summary}, handle,
                  indent=2, sort_keys=True, default=str)
    print("\nwrote {} and {}".format(output / "per_video.csv", output / "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
