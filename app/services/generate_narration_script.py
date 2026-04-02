from __future__ import annotations

import json
from typing import Dict, List

from loguru import logger

from app.services.timeline_allocator import estimate_char_budget, fit_check, trim_text_to_budget
from app.utils import utils

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from app.services.llm.migration_adapter import generate_narration as legacy_generate_narration
except Exception:  # pragma: no cover
    legacy_generate_narration = None


STYLE_GUIDE = {
    "documentary": "客观、清楚、偏影视旁白口吻，避免夸张。",
    "short_drama": "更有悬念感和戏剧张力，但不能改事实。",
    "general": "自然口语化，信息清楚，像真人解说。",
    "default": "自然口语化，信息清楚，避免空话。",
}

FACTS_PROMPT = """
你是影视事实记录员。严格基于证据写一句解说，不要脑补。

证据包：
{evidence_json}

剧情理解：
{understanding_json}

全局摘要：
{global_summary}

要求：
1. 只写本剧情块已经能确认的事实
2. 不得虚构人物动机、道具、地点
3. 如果是过渡块，就一句带过；如果是高重要度块，可写得更完整
4. 控制在 {char_budget} 字内
5. 只输出一句中文，不要解释
""".strip()

POLISH_PROMPT = """
你是影视解说文案编辑。把下面这句事实解说改成更适合口播的版本。

事实句：{fact_narration}
风格：{style_guide}
情绪：{emotion}
叙事策略：{narrative_strategy}
字数上限：{char_budget}

要求：
1. 不改事实、不改时间顺序
2. 可以更顺口、更有钩子，但不要空泛
3. 如果证据置信度较低，请用保守表达
4. 只输出一句中文，不要解释
""".strip()


def generate_narration(markdown_content: str, api_key: str, base_url: str, model: str) -> str:
    if legacy_generate_narration:
        return legacy_generate_narration(markdown_content, api_key, base_url, model)
    return ""


def _call_chat_completion(prompt: str, api_key: str, base_url: str, model: str) -> str:
    if not requests or not api_key or not base_url or not model:
        return ""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": "你是一个严格、可靠的中文影视解说助手。"},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.warning("LLM 调用失败，回退规则文案: {}", exc)
        return ""


def _fallback_fact(pkg: Dict, char_budget: int) -> str:
    text = (pkg.get("main_text_evidence") or pkg.get("subtitle_text") or pkg.get("aligned_subtitle_text") or "").strip()
    und = pkg.get("local_understanding") or {}
    protagonist_related = pkg.get("protagonist_related")
    if und.get("core_event"):
        base = und["core_event"]
        if protagonist_related:
            base = f"主角这边，{base}"
        return trim_text_to_budget(base, char_budget)
    if text:
        return trim_text_to_budget(text, char_budget)
    visual = pkg.get("visual_summary") or []
    if visual:
        desc = "，".join(x.get("desc", "") for x in visual[:2] if x.get("desc"))
        if desc:
            return trim_text_to_budget(desc, char_budget)
    return trim_text_to_budget("这一段的关键信息已经出现。", char_budget)


def _fallback_polish(fact: str, pkg: Dict, style: str, char_budget: int) -> str:
    emotion = pkg.get("emotion_hint") or "平静"
    level = pkg.get("narration_level") or "standard"
    role = pkg.get("plot_role") or "development"
    text = fact
    if style == "short_drama":
        if level == "focus":
            text = f"关键转折来了，{fact}"
        elif role == "ending":
            text = f"故事来到最后，{fact}"
        else:
            text = f"镜头一转，{fact}"
    elif style == "documentary":
        text = fact
    if emotion in {"惊讶", "紧张"} and not text.startswith("此时"):
        text = f"此时，{text}"
    return trim_text_to_budget(text, char_budget)


def _pick_ost(pkg: Dict) -> int:
    strategy = pkg.get("audio_strategy") or "duck"
    if strategy == "keep":
        return 1
    if pkg.get("block_type") == "visual" and pkg.get("importance_level") == "high":
        return 1
    return 2


def _build_script_item(idx: int, pkg: Dict, narration: str, picture: str, global_summary: Dict) -> Dict:
    start = float(pkg.get("start", pkg.get("time_window", [0.0, 0.0])[0]) or 0.0)
    end = float(pkg.get("end", pkg.get("time_window", [0.0, 0.0])[1]) or 0.0)
    if end <= start:
        end = start + 1.0
    duration = end - start
    planned_budget = int(pkg.get("planned_char_budget") or 0)
    base_budget = estimate_char_budget(duration)
    char_budget = max(planned_budget, min(base_budget + 12, int(base_budget * 1.25))) if planned_budget else base_budget
    canonical_timestamp = f"{utils.format_time(start)}-{utils.format_time(end)}"
    item = {
        "_id": idx,
        "timestamp": canonical_timestamp,
        "source_timestamp": pkg.get("timestamp") or canonical_timestamp,
        "picture": (picture or "").strip()[:60] or "画面推进",
        "narration": trim_text_to_budget((narration or "").strip(), char_budget),
        "OST": _pick_ost(pkg),
        "evidence_refs": list(pkg.get("subtitle_ids") or []) + [pkg.get("segment_id")],
        "char_budget": char_budget,
        "emotion": pkg.get("emotion_hint") or "平静",
        "segment_id": pkg.get("segment_id"),
        "scene_id": pkg.get("scene_id"),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "plot_role": pkg.get("plot_role"),
        "attraction_level": pkg.get("attraction_level"),
        "importance_level": pkg.get("importance_level"),
        "narration_level": pkg.get("narration_level"),
        "confidence": pkg.get("confidence"),
        "global_arc": global_summary.get("arc"),
    }
    item["fit_check"] = fit_check(item["narration"], duration)
    return item


def generate_narration_from_scene_evidence(
    scene_evidence: List[Dict],
    api_key: str,
    base_url: str,
    model: str,
    style: str = "documentary",
) -> List[Dict]:
    if not scene_evidence:
        return []

    global_summary = {}
    if scene_evidence and isinstance(scene_evidence[0].get("_global_summary"), dict):
        global_summary = scene_evidence[0]["_global_summary"]

    style_guide = STYLE_GUIDE.get(style, STYLE_GUIDE["default"])
    narrative_strategy = global_summary.get("narrative_strategy", "chronological")

    script_items: List[Dict] = []
    for idx, pkg in enumerate(scene_evidence, start=1):
        start = float(pkg.get("start", pkg.get("time_window", [0.0, 0.0])[0]) or 0.0)
        end = float(pkg.get("end", pkg.get("time_window", [0.0, 0.0])[1]) or 0.0)
        duration = max(end - start, 0.1)
        char_budget = int(pkg.get("planned_char_budget") or estimate_char_budget(duration))
        pkg["char_budget"] = char_budget
        evidence_json = json.dumps(
            {
                "main_text_evidence": pkg.get("main_text_evidence") or pkg.get("subtitle_text"),
                "visual_summary": pkg.get("visual_summary"),
                "context_window": pkg.get("context_window"),
                "confidence": pkg.get("confidence"),
                "emotion_hint": pkg.get("emotion_hint"),
                "entities": pkg.get("entities"),
                "block_type": pkg.get("block_type"),
                "plot_role": pkg.get("plot_role"),
                "importance_level": pkg.get("importance_level"),
                "narration_level": pkg.get("narration_level"),
            },
            ensure_ascii=False,
        )
        fact = _call_chat_completion(
            FACTS_PROMPT.format(
                evidence_json=evidence_json,
                understanding_json=json.dumps(pkg.get("local_understanding") or {}, ensure_ascii=False),
                global_summary=json.dumps(global_summary or {}, ensure_ascii=False),
                char_budget=char_budget,
            ),
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        if not fact:
            fact = _fallback_fact(pkg, char_budget)
        fact = trim_text_to_budget(fact, char_budget)

        polish_budget = max(char_budget, int(char_budget * 1.1))
        polished = _call_chat_completion(
            POLISH_PROMPT.format(
                fact_narration=fact,
                style_guide=style_guide,
                emotion=pkg.get("emotion_hint") or "平静",
                narrative_strategy=narrative_strategy,
                char_budget=polish_budget,
            ),
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        if not polished:
            polished = _fallback_polish(fact, pkg, style, polish_budget)
        if pkg.get("confidence") in {"asr_low", "visual_only"}:
            polished = polished.replace("一定", "似乎").replace("就是", "像是")
        picture = pkg.get("picture") or "；".join(
            x.get("desc", "") for x in (pkg.get("visual_summary") or [])[:2]
        )
        script_items.append(_build_script_item(idx, pkg, polished, picture, global_summary or {}))

    return script_items
