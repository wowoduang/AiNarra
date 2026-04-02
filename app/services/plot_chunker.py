from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from loguru import logger


SPLIT_CUES = (
    "突然", "这时", "另一边", "与此同时", "后来", "随后", "最后", "最终", "原来", "然而", "可是",
)
ACTION_CUES = ("打", "冲", "追", "跑", "撞", "救", "杀", "逃", "爆炸", "开枪")
EMOTION_CUES = ("哭", "笑", "怒", "害怕", "惊", "沉默", "崩溃", "委屈", "紧张")
CLIMAX_CUES = ("真相", "原来", "终于", "没想到", "居然", "结局", "最后", "反转")
TRANSITION_CUES = ("然后", "接着", "之后", "另一边", "与此同时", "第二天", "后来")


@dataclass
class ChunkPlan:
    target_duration_minutes: int = 8
    narrative_strategy: str = "chronological"
    accuracy_priority: str = "high"


def _seg_text(seg: Dict) -> str:
    return str(seg.get("text") or seg.get("content") or "").strip()


def _duration(start: float, end: float) -> float:
    return max(float(end or 0.0) - float(start or 0.0), 0.0)


def _contains_any(text: str, cues: Sequence[str]) -> bool:
    return any(c in text for c in cues)


def _guess_block_type(text: str, visual_only: bool = False) -> str:
    if visual_only:
        return "visual"
    if _contains_any(text, ACTION_CUES):
        return "action"
    if _contains_any(text, EMOTION_CUES):
        return "emotion"
    if _contains_any(text, TRANSITION_CUES):
        return "transition"
    return "dialogue"


def _guess_plot_role(index: int, total: int, text: str) -> str:
    if total <= 1:
        return "setup"
    pos = index / max(total - 1, 1)
    if pos <= 0.15:
        return "setup"
    if pos >= 0.82:
        return "ending"
    if _contains_any(text, CLIMAX_CUES):
        return "twist"
    if _contains_any(text, ACTION_CUES + EMOTION_CUES):
        return "conflict"
    return "development"


def _importance_score(text: str, role: str, duration: float) -> int:
    score = 1
    if _contains_any(text, CLIMAX_CUES):
        score += 3
    if _contains_any(text, ACTION_CUES):
        score += 2
    if _contains_any(text, EMOTION_CUES):
        score += 1
    if role in {"twist", "ending"}:
        score += 2
    if duration > 25:
        score += 1
    return score


def _importance_level(score: int) -> str:
    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _frame_budget(block_type: str, importance: str, duration: float) -> int:
    if block_type == "visual":
        return 3
    if importance == "high":
        return 3 if duration >= 18 else 2
    if importance == "medium":
        return 2 if duration >= 12 else 1
    return 1


def _audio_strategy(block_type: str, importance: str) -> str:
    if block_type in {"action", "visual"} and importance != "low":
        return "keep"
    return "duck"


def _build_candidate_times(start: float, end: float, count: int) -> List[float]:
    count = max(int(count or 1), 1)
    duration = max(end - start, 0.2)
    if count == 1:
        return [round(start + duration * 0.5, 3)]
    points = []
    for i in range(count):
        ratio = (i + 1) / (count + 1)
        points.append(round(start + duration * ratio, 3))
    return points


def _should_split(prev_seg: Dict, seg: Dict, bucket: List[Dict], bucket_text: str) -> bool:
    if not bucket:
        return False
    gap = float(seg.get("start", 0.0)) - float(prev_seg.get("end", 0.0))
    if gap > 2.8:
        return True
    text = _seg_text(seg)
    total_duration = float(seg.get("end", 0.0)) - float(bucket[0].get("start", 0.0))
    total_chars = len(bucket_text) + len(text)
    if total_duration >= 55:
        return True
    if total_chars >= 260:
        return True
    if len(bucket) >= 8 and _contains_any(text, SPLIT_CUES):
        return True
    if _contains_any(text, CLIMAX_CUES) and total_duration >= 15:
        return True
    return False


def _merge_micro_chunks(chunks: List[Dict]) -> List[Dict]:
    if len(chunks) <= 1:
        return chunks
    merged: List[Dict] = [dict(chunks[0])]
    for chunk in chunks[1:]:
        duration = _duration(chunk["start"], chunk["end"])
        text = chunk.get("aligned_subtitle_text", "")
        if duration < 4.0 or len(text) < 14:
            prev = merged[-1]
            prev["end"] = chunk["end"]
            prev["subtitle_ids"].extend(chunk.get("subtitle_ids", []))
            prev["subtitle_texts"].extend(chunk.get("subtitle_texts", []))
            prev["aligned_subtitle_text"] = " ".join(x for x in prev["subtitle_texts"] if x).strip()
        else:
            merged.append(dict(chunk))
    return merged


def build_plot_chunks_from_subtitles(
    subtitle_segments: Sequence[Dict],
    target_duration_minutes: int = 8,
    narrative_strategy: str = "chronological",
    accuracy_priority: str = "high",
) -> List[Dict]:
    segments = [dict(x) for x in (subtitle_segments or []) if _seg_text(x)]
    if not segments:
        return []

    chunks_raw: List[Dict] = []
    bucket: List[Dict] = []
    bucket_texts: List[str] = []
    prev_seg = None
    for seg in segments:
        text = _seg_text(seg)
        if prev_seg is not None and _should_split(prev_seg, seg, bucket, " ".join(bucket_texts)):
            chunks_raw.append(_flush_bucket(bucket, bucket_texts))
            bucket = []
            bucket_texts = []
        bucket.append(seg)
        bucket_texts.append(text)
        prev_seg = seg
    if bucket:
        chunks_raw.append(_flush_bucket(bucket, bucket_texts))

    chunks_raw = _merge_micro_chunks(chunks_raw)
    total = len(chunks_raw)
    for idx, chunk in enumerate(chunks_raw, start=1):
        text = chunk.get("aligned_subtitle_text", "")
        duration = _duration(chunk["start"], chunk["end"])
        role = _guess_plot_role(idx - 1, total, text)
        block_type = _guess_block_type(text)
        score = _importance_score(text, role, duration)
        importance = _importance_level(score)
        frame_budget = _frame_budget(block_type, importance, duration)
        chunk.update({
            "scene_id": f"plot_{idx:03d}",
            "segment_id": f"plot_{idx:03d}",
            "plot_role": role,
            "block_type": block_type,
            "importance_score": score,
            "importance_level": importance,
            "attraction_level": "高" if importance == "high" else ("中" if importance == "medium" else "低"),
            "audio_strategy": _audio_strategy(block_type, importance),
            "frame_budget": frame_budget,
            "keyframe_candidates": _build_candidate_times(chunk["start"], chunk["end"], frame_budget),
            "target_duration_minutes": int(target_duration_minutes or 8),
            "narrative_strategy": narrative_strategy or "chronological",
            "accuracy_priority": accuracy_priority or "high",
            "visual_only": False,
        })
    planned = apply_duration_plan(chunks_raw, ChunkPlan(target_duration_minutes, narrative_strategy, accuracy_priority))
    logger.info("剧情块构建完成: {} 个剧情块, target_minutes={}", len(planned), target_duration_minutes)
    return planned


def _flush_bucket(bucket: List[Dict], texts: List[str]) -> Dict:
    subtitle_ids = [str(x.get("seg_id") or x.get("id") or "") for x in bucket]
    start = float(bucket[0].get("start", 0.0))
    end = float(bucket[-1].get("end", start + 1.0))
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "subtitle_ids": subtitle_ids,
        "subtitle_texts": texts[:],
        "aligned_subtitle_text": " ".join(x for x in texts if x).strip(),
        "subtitle_source": bucket[0].get("subtitle_source", "generated_srt"),
    }


def apply_duration_plan(chunks: Sequence[Dict], plan: ChunkPlan) -> List[Dict]:
    items = [dict(x) for x in (chunks or [])]
    if not items:
        return []
    movie_minutes = max(float(items[-1].get("end", 0.0)) / 60.0, 1.0)
    target_minutes = max(int(plan.target_duration_minutes or 8), 1)
    ratio = target_minutes / movie_minutes

    for item in items:
        importance = item.get("importance_level", "medium")
        role = item.get("plot_role", "development")
        if ratio <= 0.08:
            level = "focus" if importance == "high" or role in {"ending", "twist"} else "brief"
        elif ratio <= 0.14:
            level = "focus" if importance == "high" else ("standard" if importance == "medium" else "brief")
        else:
            level = "standard" if importance != "low" else "brief"
        if plan.accuracy_priority == "high" and item.get("block_type") == "transition":
            level = "brief"
        item["narration_level"] = level

        duration = _duration(item["start"], item["end"])
        base_chars = max(int(duration * 3.0), 10)
        if level == "focus":
            planned_chars = max(int(base_chars * 1.15), 26)
        elif level == "standard":
            planned_chars = max(base_chars, 18)
        else:
            planned_chars = max(int(base_chars * 0.55), 10)
        # 软控制：不追求严格命中字数上限
        item["planned_char_budget"] = planned_chars
    return items
