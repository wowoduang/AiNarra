from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

from loguru import logger

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

NAME_PATTERNS = [
    re.compile(r"[《“\"]([^》”\"]{1,10})[》”\"]"),
    re.compile(r"\b[A-Z][a-z]{1,20}\b"),
]
EMOTION_CUES = {
    "愤怒": ["怒", "吼", "滚", "闭嘴", "骂"],
    "悲伤": ["哭", "泪", "难过", "别走", "对不起"],
    "喜悦": ["笑", "高兴", "终于", "太好了"],
    "紧张": ["快", "危险", "小心", "糟了"],
    "惊讶": ["什么", "怎么会", "居然", "没想到"],
    "恐惧": ["别过来", "救命", "害怕", "不要"],
}


def _call_chat_completion(prompt: str, api_key: str = "", base_url: str = "", model: str = "") -> str:
    if not (requests and api_key and base_url and model):
        return ""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是严格、保守的中文剧情分析助手。"},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.warning("剧情理解 LLM 调用失败，回退规则摘要: {}", exc)
        return ""


def _extract_names(text: str) -> List[str]:
    found: List[str] = []
    for pattern in NAME_PATTERNS:
        for item in pattern.findall(text or ""):
            if item and item not in found:
                found.append(item)
    return found[:8]


def _key_dialogues(text: str, max_items: int = 2) -> List[str]:
    if not text:
        return []
    pieces = re.split(r"[。！？!?\n]", text)
    out = []
    for piece in pieces:
        piece = piece.strip(" ，,；;：:")
        if len(piece) >= 4:
            out.append(piece[:28])
        if len(out) >= max_items:
            break
    return out


def _emotion(text: str, visual: List[Dict] | None = None) -> str:
    joined = (text or "") + " " + " ".join(x.get("desc", "") for x in (visual or []))
    for label, cues in EMOTION_CUES.items():
        if any(c in joined for c in cues):
            return label
    return "平静"


def _core_event(text: str) -> str:
    dialogs = _key_dialogues(text, max_items=1)
    if dialogs:
        return dialogs[0][:20]
    return (text or "剧情继续推进")[:20]


def _heuristic_global_summary(items: Sequence[Dict]) -> Dict:
    full_text = " ".join((x.get("aligned_subtitle_text") or x.get("subtitle_text") or x.get("main_text_evidence") or "") for x in items)
    names = _extract_names(full_text)
    protagonist = names[0] if names else "主角"
    key_segments = [x.get("segment_id") for x in items if x.get("importance_level") == "high"][:10]
    main_events = []
    for item in items[:8]:
        text = (item.get("aligned_subtitle_text") or item.get("subtitle_text") or item.get("main_text_evidence") or "").strip()
        if text:
            main_events.append(_core_event(text))
    return {
        "protagonist": protagonist,
        "main_storyline": "；".join(main_events[:4])[:90] or "故事围绕主角的冲突与转折推进。",
        "character_relations": [{"a": protagonist, "relation": "关联人物", "b": n} for n in names[1:4]],
        "unresolved_tensions": _key_dialogues(full_text, max_items=5),
        "entity_map": {n: n for n in names},
        "arc": items[-1].get("plot_role", "development") if items else "development",
        "key_segments": [x for x in key_segments if x],
    }


def build_global_summary(
    evidence_list: List[Dict],
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Dict:
    if not evidence_list:
        return {
            "protagonist": "主角",
            "main_storyline": "",
            "character_relations": [],
            "unresolved_tensions": [],
            "entity_map": {},
            "arc": "unknown",
            "key_segments": [],
        }

    summary = _heuristic_global_summary(evidence_list)
    # 尝试让 LLM 把全剧理解再压成更像电影解说的全局摘要
    text_lines = []
    for item in evidence_list[:40]:
        text = (item.get("aligned_subtitle_text") or item.get("subtitle_text") or item.get("main_text_evidence") or "").strip()
        if not text:
            continue
        text_lines.append(
            f"[{item.get('segment_id')}] {item.get('plot_role','development')} {item.get('importance_level','medium')}: {text[:120]}"
        )
    prompt = (
        "请基于以下字幕剧情块，输出 JSON：\n"
        + '{"protagonist":"...","main_storyline":"...","character_relations":[...],"unresolved_tensions":[...],"key_segments":[...]}\n'
        + "不要脑补画面，只根据剧情块做整剧理解。\n\n"
        + "\n".join(text_lines)
    )
    raw = _call_chat_completion(prompt, api_key=api_key, base_url=base_url, model=model)
    if raw:
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    summary.update({k: v for k, v in data.items() if v is not None})
        except Exception:
            logger.warning("解析 LLM 全局剧情摘要失败，继续使用规则摘要")
    logger.info("全局剧情理解完成: protagonist={}, key_segments={}", summary.get("protagonist"), len(summary.get("key_segments", [])))
    return summary


def add_local_understanding(evidence_list: List[Dict]) -> List[Dict]:
    if not evidence_list:
        return []
    global_summary = evidence_list[0].get("_global_summary") if isinstance(evidence_list[0].get("_global_summary"), dict) else {}
    protagonist = global_summary.get("protagonist") or "主角"
    for pkg in evidence_list:
        text = (pkg.get("subtitle_text") or pkg.get("main_text_evidence") or pkg.get("aligned_subtitle_text") or "").strip()
        chars = _extract_names(text)
        understanding = {
            "characters": chars,
            "core_event": _core_event(text),
            "key_dialogue": _key_dialogues(text),
            "conflict_or_twist": text[:24] if pkg.get("plot_role") in {"conflict", "twist", "ending"} else None,
            "emotion": _emotion(text, pkg.get("visual_summary") or []),
        }
        pkg["local_understanding"] = understanding
        pkg["emotion_hint"] = pkg.get("emotion_hint") or understanding["emotion"]
        pkg["protagonist_related"] = protagonist in chars or pkg.get("importance_level") == "high"
        if pkg["protagonist_related"] and pkg.get("narration_level") == "brief":
            pkg["narration_level"] = "standard"
    return evidence_list
