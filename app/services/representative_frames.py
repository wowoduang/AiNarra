from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

from loguru import logger

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False


def extract_representative_frames_for_scenes(
    video_path: str,
    scenes: List[Dict],
    *,
    visual_mode: str = "auto",
    output_dir: str = "",
    max_frames_dialogue: int = 1,
    max_frames_visual_only: int = 3,
    max_frames_long_scene: int = 3,
    long_scene_threshold: float = 25.0,
    jpeg_quality: int = 80,
    max_edge: int = 960,
) -> List[Dict]:
    if visual_mode == "off":
        logger.info("代表帧抽取关闭: visual_mode=off")
        return []
    if not _CV2_AVAILABLE:
        logger.warning("OpenCV 不可用，跳过代表帧抽取")
        return []
    if not video_path or not os.path.isfile(video_path):
        logger.warning("视频文件不存在，跳过代表帧抽取")
        return []
    if not scenes:
        return []
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "videaai_story_frames")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("无法打开视频，跳过代表帧抽取")
        return []

    records: List[Dict] = []
    try:
        for scene in scenes:
            scene_id = scene.get("scene_id") or scene.get("segment_id") or "scene"
            start = float(scene.get("start", 0.0) or 0.0)
            end = float(scene.get("end", start + 1.0) or (start + 1.0))
            duration = max(end - start, 0.2)
            budget = int(scene.get("frame_budget") or 0)
            if budget <= 0:
                if scene.get("visual_only"):
                    budget = max_frames_visual_only
                elif duration >= long_scene_threshold or scene.get("importance_level") == "high":
                    budget = max_frames_long_scene
                else:
                    budget = max_frames_dialogue
            budget = max(1, min(budget, 6))
            timestamps = _candidate_times(start, end, budget)
            for rank, ts in enumerate(timestamps, start=1):
                frame = _seek_frame(cap, ts)
                if frame is None:
                    continue
                frame = _resize_keep_ratio(frame, max_edge=max_edge)
                file_path = os.path.join(output_dir, f"{scene_id}_{rank:02d}_{ts:.3f}.jpg")
                cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                records.append({
                    "scene_id": scene_id,
                    "segment_id": scene.get("segment_id") or scene_id,
                    "frame_path": file_path,
                    "timestamp_seconds": round(ts, 3),
                    "rank": rank,
                })
    finally:
        cap.release()
    logger.info("代表帧抽取完成: {} 张", len(records))
    return records


def _candidate_times(start: float, end: float, count: int) -> List[float]:
    duration = max(end - start, 0.2)
    if count <= 1:
        return [round(start + duration * 0.5, 3)]
    points = []
    for idx in range(count):
        ratio = (idx + 1) / (count + 1)
        points.append(round(start + duration * ratio, 3))
    return points


def _seek_frame(cap, ts: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(ts, 0.0) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _resize_keep_ratio(frame, max_edge: int = 960):
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return frame
    scale = max_edge / float(long_edge)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return cv2.resize(frame, (new_w, new_h))
