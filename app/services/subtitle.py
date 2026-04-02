import os
import os.path
import re
import sys
import traceback
import subprocess
import json
from typing import Optional, List, Tuple, Dict, Any

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from funasr import AutoModel as FunASRAutoModel
except Exception:
    FunASRAutoModel = None

from timeit import default_timer as timer
from loguru import logger
import google.generativeai as genai
from moviepy import VideoFileClip

from app.config import config
from app.utils import utils


model_size = config.whisper.get("model_size", "faster-whisper-large-v3")
backend_name = str(config.whisper.get("backend", "")).strip().lower()
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")
model = None
_model_identity = None


DEFAULT_AUDIO_SAMPLE_RATE = int(config.whisper.get("audio_sample_rate", 16000))
DEFAULT_AUDIO_CHANNELS = int(config.whisper.get("audio_channels", 1))
DEFAULT_AUDIO_BITRATE = str(config.whisper.get("audio_bitrate", "32k"))
DEFAULT_FORCE_LANGUAGE = str(config.whisper.get("language", config.ui.get("language", "zh"))).strip() or None
DEFAULT_BEAM_SIZE = int(config.whisper.get("beam_size", 1 if str(device).lower() == "cpu" else 5))
DEFAULT_BEST_OF = int(config.whisper.get("best_of", 1 if str(device).lower() == "cpu" else 3))
DEFAULT_VAD_MIN_SILENCE_MS = int(config.whisper.get("vad_min_silence_duration_ms", 400))
DEFAULT_INITIAL_PROMPT = config.whisper.get("initial_prompt", "以下是普通话的人声内容，请准确转写字幕。")
DEFAULT_AUDIO_FILTER = config.whisper.get(
    "audio_filter",
    "highpass=f=120,lowpass=f=3800,volume=1.8",
)
DEFAULT_FUNASR_BATCH_SIZE_S = int(config.whisper.get("funasr_batch_size_s", 180))
DEFAULT_FUNASR_MERGE_LENGTH_S = int(config.whisper.get("funasr_merge_length_s", 15))
DEFAULT_FUNASR_USE_ITN = bool(config.whisper.get("funasr_use_itn", True))
DEFAULT_FUNASR_VAD_SEGMENT_MS = int(config.whisper.get("funasr_max_single_segment_time_ms", 30000))
DEFAULT_FUNASR_VAD_MODEL = str(config.whisper.get("funasr_vad_model", "")).strip()


MODEL_CANDIDATES = {
    "sensevoice": [
        "SenseVoiceSmall",
        "SenseVoiceSmall-onnx",
        "iic/SenseVoiceSmall",
    ],
    "faster-whisper": [
        "faster-whisper-large-v3",
        "faster-whisper-large-v2",
        "large-v3",
        "large-v2",
    ],
    "funasr-vad": [
        "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "speech_fsmn_vad_zh-cn-16k-common",
        "fsmn-vad",
    ],
}

_MODEL_FILES = ("model.bin", "model.safetensors")

# ---------- subtitle cleaning / normalization ----------

_CONTROL_TAG_GROUP_RE = re.compile(r"(?:<\|[^|]+?\|>\s*)+")
_CONTROL_TAG_RE = re.compile(r"<\|[^|]+?\|>")
_ASS_TAG_RE = re.compile(r"\{\\[^{}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
_DUP_PUNC_RE = re.compile(r"([，。！？；：,.!?;:])\1{1,}")

_GARBAGE_WORDS_RE = re.compile(
    r"\b(?:Speech|BGM|EMO_UNKNOWN|NEUTRAL|HAPPY|ANGRY|SAD|FEAR|SURPRISE|DISGUST|withitn|withit|withint)\b",
    re.IGNORECASE,
)

_CUE_ONLY_PATTERNS = [
    r"^\(?\s*(bgm|music|applause|laughter|laughing|sigh|crying|phone ringing|ringtone)\s*\)?$",
    r"^\(?\s*(音乐|配乐|背景音乐|掌声|笑声|哭声|叹气|铃声|电话铃声|风声|雨声|脚步声|鼓掌)\s*\)?$",
    r"^\[?\s*(bgm|music|applause|laughter|laughing|sigh|crying)\s*\]?$",
    r"^\[?\s*(音乐|配乐|背景音乐|掌声|笑声|哭声)\s*\]?$",
]
_CUE_ONLY_RE = re.compile("|".join(_CUE_ONLY_PATTERNS), re.IGNORECASE)

_SEGMENT_JSON_SUFFIX = "_segments.json"
_RAW_SRT_SUFFIX = "_raw.srt"
_CLEAN_SRT_SUFFIX = "_clean.srt"


def _base_dirs() -> List[str]:
    seen: set = set()
    bases: List[str] = []
    for raw in (
        utils.root_dir(),
        os.getcwd(),
        os.path.dirname(os.path.abspath(sys.executable)),
    ):
        d = os.path.abspath(raw)
        if d not in seen:
            seen.add(d)
            bases.append(d)
    return bases


def _normalize_backend() -> str:
    if backend_name:
        if "sensevoice" in backend_name:
            return "sensevoice"
        if "paraformer" in backend_name or "funasr" in backend_name:
            return "funasr"
        if "whisper" in backend_name:
            return "faster-whisper"
        return backend_name

    lowered = str(model_size or "").strip().lower()
    if "sensevoice" in lowered:
        return "sensevoice"
    if "paraformer" in lowered:
        return "funasr"
    return "faster-whisper"


CURRENT_BACKEND = _normalize_backend()


def _derive_debug_paths(subtitle_file: str) -> Tuple[str, str]:
    root, ext = os.path.splitext(subtitle_file)
    if not ext:
        ext = ".srt"
    return f"{root}{_RAW_SRT_SUFFIX}", f"{root}{_CLEAN_SRT_SUFFIX}"


def _derive_segments_json_path(subtitle_file: str) -> str:
    root, _ = os.path.splitext(subtitle_file)
    return f"{root}{_SEGMENT_JSON_SUFFIX}"


def _clean_subtitle_text(text: str) -> str:
    text = text or ""
    text = _CONTROL_TAG_GROUP_RE.sub("\n", text)
    text = _ASS_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _CONTROL_TAG_RE.sub("", text)
    text = _GARBAGE_WORDS_RE.sub("", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[·•]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\s*([，。！？；：、“”‘’《》、,.!?;:])\s*", r"\1", text)
    text = _DUP_PUNC_RE.sub(r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"（\s*）", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    return text.strip()


def _is_meaningful_subtitle_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if _CUE_ONLY_RE.match(stripped):
        return False
    if not re.sub(r"[，。！？；：、“”‘’《》、,.!?;:\-\—~…·\s]", "", stripped):
        return False
    return True


def _extract_clean_sentence_units(text: str) -> List[str]:
    cleaned = _clean_subtitle_text(text)
    if not cleaned:
        return []

    coarse_parts = [p.strip() for p in cleaned.split("\n") if p.strip()]
    final_parts: List[str] = []

    for part in coarse_parts:
        if not _is_meaningful_subtitle_text(part):
            continue

        strong_parts = [seg.strip() for seg in re.split(r"(?<=[。！？!?；;])", part) if seg.strip()]
        refined: List[str] = []

        for seg in strong_parts:
            if len(seg) > 42:
                comma_parts = [x.strip() for x in re.split(r"(?<=[，,])", seg) if x.strip()]
                refined.extend(comma_parts or [seg])
            else:
                refined.append(seg)

        for seg in refined:
            seg = _clean_subtitle_text(seg)
            if _is_meaningful_subtitle_text(seg):
                final_parts.append(seg)

    return final_parts


def _is_valid_model_dir(path: str) -> bool:
    return any(os.path.isfile(os.path.join(path, f)) for f in _MODEL_FILES)


def _candidate_model_dirs(names: List[str]) -> List[str]:
    dirs: List[str] = []
    seen: set = set()
    for base in _base_dirs():
        for name in names:
            if not name or "/" in name:
                continue
            p = os.path.abspath(os.path.join(base, "app", "models", name))
            if p not in seen:
                seen.add(p)
                dirs.append(p)
    return dirs


def _resolve_local_model_path(names: List[str], *, require_ctranslate2: bool = False) -> Tuple[Optional[str], List[str]]:
    searched = _candidate_model_dirs(names)
    for path in searched:
        if os.path.isdir(path):
            if not require_ctranslate2 or _is_valid_model_dir(path):
                return path, searched
            try:
                contents = os.listdir(path)
            except OSError:
                contents = []
            logger.warning(
                f"模型目录存在但未找到模型文件 ({', '.join(_MODEL_FILES)}): {path}\n目录内容: {contents}"
            )
    return None, searched


def _resolve_faster_whisper_model_path() -> Tuple[Optional[str], List[str]]:
    requested = (str(model_size or "").strip() or MODEL_CANDIDATES["faster-whisper"][0])
    candidates = [requested]
    for fallback in MODEL_CANDIDATES["faster-whisper"]:
        if fallback not in candidates:
            candidates.append(fallback)
    return _resolve_local_model_path(candidates, require_ctranslate2=True)


def _resolve_sensevoice_model_ref() -> Tuple[str, List[str]]:
    requested = str(model_size or "").strip()
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(MODEL_CANDIDATES["sensevoice"])
    local_path, searched = _resolve_local_model_path(candidates, require_ctranslate2=False)
    if local_path:
        return local_path, searched
    if requested and os.path.isdir(requested):
        return requested, searched
    if requested and "/" in requested:
        return requested, searched
    return "iic/SenseVoiceSmall", searched


def _resolve_funasr_vad_model_ref() -> Tuple[str, List[str]]:
    requested = DEFAULT_FUNASR_VAD_MODEL
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(MODEL_CANDIDATES["funasr-vad"])
    local_path, searched = _resolve_local_model_path(candidates, require_ctranslate2=False)
    if local_path:
        return local_path, searched
    if requested:
        return requested, searched
    return "fsmn-vad", searched


def _load_faster_whisper_model() -> bool:
    global model, device, compute_type, _model_identity
    if WhisperModel is None:
        logger.error("未安装 faster-whisper，请先安装依赖后再使用自动字幕生成")
        return False

    model_path, searched = _resolve_faster_whisper_model_path()
    if not model_path:
        searched_text = "\n".join([f"- {p}" for p in searched])
        bases_text = "\n".join([f"- {d}" for d in _base_dirs()])
        logger.error(
            "请先下载 whisper 模型\n\n"
            "********************************************\n"
            "推荐下载：\n"
            "https://huggingface.co/guillaumekln/faster-whisper-large-v3\n"
            "或：\n"
            "https://huggingface.co/guillaumekln/faster-whisper-large-v2\n\n"
            "存放路径示例：app/models/faster-whisper-large-v3\n\n"
            f"root_dir(): {utils.root_dir()}\n"
            f"cwd: {os.getcwd()}\n"
            f"executable: {sys.executable}\n\n"
            "搜索基础目录：\n"
            f"{bases_text}\n\n"
            "已搜索目录：\n"
            f"{searched_text}\n"
            "********************************************\n"
        )
        return False

    identity = f"faster-whisper::{model_path}"
    if model is not None and _model_identity == identity:
        return True

    use_cuda = False
    try:
        def check_cuda_available():
            try:
                import torch
                return torch.cuda.is_available()
            except Exception as e:
                logger.warning(f"检查CUDA可用性时出错: {e}")
                return False

        use_cuda = check_cuda_available()
        if use_cuda:
            logger.info(f"尝试使用 CUDA 加载 faster-whisper 模型: {model_path}")
            try:
                model = WhisperModel(
                    model_size_or_path=model_path,
                    device="cuda",
                    compute_type="float16",
                    local_files_only=True,
                )
                device = "cuda"
                compute_type = "float16"
                _model_identity = identity
                logger.info("成功使用 CUDA 加载 faster-whisper 模型")
                return True
            except Exception as e:
                logger.warning(f"CUDA 加载 faster-whisper 失败，回退到 CPU: {e}")
                use_cuda = False
    except Exception as e:
        logger.warning(f"CUDA检查过程出错，默认使用CPU: {e}")
        use_cuda = False

    device = "cpu"
    compute_type = "int8"
    logger.info(f"使用 CPU 加载 faster-whisper 模型: {model_path}")
    model = WhisperModel(
        model_size_or_path=model_path,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    _model_identity = identity
    logger.info(f"模型加载完成，使用设备: {device}, 计算类型: {compute_type}")
    return True


def _sensevoice_device() -> str:
    prefer = str(config.whisper.get("device", device)).strip().lower() or "cpu"
    if prefer in {"cuda", "gpu"}:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def _load_sensevoice_model() -> bool:
    global model, _model_identity
    if FunASRAutoModel is None:
        logger.error("未安装 funasr，请先安装 funasr 和 modelscope 后再使用 SenseVoice-Small")
        return False

    model_ref, searched = _resolve_sensevoice_model_ref()
    vad_ref, vad_searched = _resolve_funasr_vad_model_ref()
    identity = f"sensevoice::{model_ref}::{vad_ref}"
    if model is not None and _model_identity == identity:
        return True

    logger.info(
        f"加载 SenseVoice-Small: model={model_ref}, vad_model={vad_ref}, device={_sensevoice_device()}"
    )
    try:
        model = FunASRAutoModel(
            model=model_ref,
            vad_model=vad_ref,
            vad_kwargs={"max_single_segment_time": DEFAULT_FUNASR_VAD_SEGMENT_MS},
            device=_sensevoice_device(),
            disable_progress_bar=True,
        )
        _model_identity = identity
        return True
    except Exception as e:
        logger.error(
            "加载 SenseVoice-Small 失败。\n"
            f"model={model_ref}\n"
            f"vad_model={vad_ref}\n"
            f"searched_models={searched}\n"
            f"searched_vad={vad_searched}\n"
            f"error={e}"
        )
        return False


def _load_model() -> bool:
    backend = CURRENT_BACKEND
    if backend == "sensevoice":
        return _load_sensevoice_model()
    if backend == "funasr":
        return _load_sensevoice_model()
    return _load_faster_whisper_model()


def _normalize_language_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = str(lang).strip().lower()
    mapping = {
        "zh-cn": "zh",
        "zh_simplified": "zh",
        "zh-hans": "zh",
        "cn": "zh",
        "chs": "zh",
        "zn": "zh",
    }
    return mapping.get(lang, lang)


def _safe_file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def _extract_audio_ffmpeg(video_file: str, audio_file: str) -> bool:
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-vn",
        "-map", "a:0",
        "-ac", str(DEFAULT_AUDIO_CHANNELS),
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-af", DEFAULT_AUDIO_FILTER,
        "-c:a", "libmp3lame",
        "-b:a", DEFAULT_AUDIO_BITRATE,
        audio_file,
    ]
    logger.info(
        f"使用 ffmpeg 提取人声音频: sample_rate={DEFAULT_AUDIO_SAMPLE_RATE}, "
        f"channels={DEFAULT_AUDIO_CHANNELS}, bitrate={DEFAULT_AUDIO_BITRATE}, filter={DEFAULT_AUDIO_FILTER}"
    )
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and os.path.exists(audio_file):
        logger.info(f"ffmpeg 音频提取完成: {audio_file} ({_safe_file_size_mb(audio_file):.2f} MB)")
        return True
    logger.warning(f"ffmpeg 提取音频失败，准备回退 moviepy: {proc.stderr[-1200:] if proc.stderr else 'unknown error'}")
    return False


def _append_subtitle_line(subtitles: List[Dict[str, Any]], seg_text: str, seg_start: float, seg_end: float):
    seg_text = _clean_subtitle_text(seg_text or "")
    if not _is_meaningful_subtitle_text(seg_text):
        return
    seg_start = max(0.0, float(seg_start or 0.0))
    seg_end = max(seg_start, float(seg_end or seg_start))
    subtitles.append({"msg": seg_text, "start_time": seg_start, "end_time": seg_end})


def _append_raw_subtitle_line(raw_subtitles: List[Dict[str, Any]], seg_text: str, seg_start: float, seg_end: float):
    seg_text = (seg_text or "").strip()
    if not seg_text:
        return
    seg_start = max(0.0, float(seg_start or 0.0))
    seg_end = max(seg_start, float(seg_end or seg_start))
    raw_subtitles.append({"msg": seg_text, "start_time": seg_start, "end_time": seg_end})


def _coerce_seconds(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    return num / 1000.0 if num > 1000 else num


def _split_sentences_keep_punctuation(text: str) -> List[str]:
    parts = _extract_clean_sentence_units(text)
    if parts:
        return parts

    text = _clean_subtitle_text(text)
    if not _is_meaningful_subtitle_text(text):
        return []

    parts = [seg.strip() for seg in re.split(r"(?<=[。！？!?；;])\s*", text) if seg.strip()]
    if len(parts) <= 1:
        parts = [seg.strip() for seg in re.split(r"(?<=[，,])\s*", text) if seg.strip()]
    if len(parts) <= 1:
        parts = utils.split_string_by_punctuations(text)
    return [seg.strip() for seg in parts if _is_meaningful_subtitle_text(seg)] or ([text] if _is_meaningful_subtitle_text(text) else [])


def _append_split_subtitle_lines(subtitles: List[Dict[str, Any]], text: str, start: float, end: float) -> int:
    text = _clean_subtitle_text(text or "")
    if not _is_meaningful_subtitle_text(text):
        return 0

    start = max(0.0, float(start or 0.0))
    end = max(start, float(end or start))
    sentences = _split_sentences_keep_punctuation(text)
    if not sentences:
        return 0
    if len(sentences) <= 1:
        _append_subtitle_line(subtitles, text, start, end)
        return 1

    total_chars = sum(max(1, len(s)) for s in sentences)
    duration = max(0.0, end - start)
    if duration <= 0:
        duration = max(1.5, 2.6 * len(sentences))
        end = start + duration

    cursor = start
    appended = 0
    for idx, sentence in enumerate(sentences):
        weight = max(1, len(sentence)) / total_chars
        if idx == len(sentences) - 1:
            seg_end = end
        else:
            seg_end = min(end, cursor + duration * weight)
        _append_subtitle_line(subtitles, sentence, cursor, seg_end)
        cursor = seg_end
        appended += 1
    return appended


def _extract_item_time_range(item: Dict[str, Any]) -> Tuple[float, float]:
    sentence_info = item.get("sentence_info") or []
    if sentence_info:
        try:
            start = _coerce_seconds(sentence_info[0].get("start", 0.0))
            end = _coerce_seconds(sentence_info[-1].get("end", start))
            return start, max(start, end)
        except Exception:
            pass
    timestamp_pairs = item.get("timestamp") or []
    if timestamp_pairs:
        try:
            return _coerce_seconds(timestamp_pairs[0][0]), max(_coerce_seconds(timestamp_pairs[0][0]), _coerce_seconds(timestamp_pairs[-1][1]))
        except Exception:
            pass
    start = _coerce_seconds(item.get("start", 0.0))
    end = _coerce_seconds(item.get("end", start))
    return start, max(start, end)


def _parse_funasr_result_item(
    item: Dict[str, Any],
    subtitles: List[Dict[str, Any]],
    raw_subtitles: Optional[List[Dict[str, Any]]] = None,
) -> int:
    count = 0
    sentence_info = item.get("sentence_info") or []
    for sentence in sentence_info:
        if not isinstance(sentence, dict):
            continue
        text = sentence.get("text") or sentence.get("sentence") or ""
        start = _coerce_seconds(sentence.get("start", 0.0))
        end = _coerce_seconds(sentence.get("end", start))
        if raw_subtitles is not None:
            _append_raw_subtitle_line(raw_subtitles, text, start, end)
        count += _append_split_subtitle_lines(subtitles, text, start, end)
    if count:
        return count

    timestamp_pairs = item.get("timestamp") or []
    text = item.get("text") or item.get("sentence") or ""
    if timestamp_pairs and text:
        try:
            start = _coerce_seconds(timestamp_pairs[0][0])
            end = _coerce_seconds(timestamp_pairs[-1][1])
            if raw_subtitles is not None:
                _append_raw_subtitle_line(raw_subtitles, text, start, end)
            return _append_split_subtitle_lines(subtitles, text, start, end)
        except Exception:
            pass

    if text:
        start, end = _extract_item_time_range(item)
        if raw_subtitles is not None:
            _append_raw_subtitle_line(raw_subtitles, text, start, end)
        return _append_split_subtitle_lines(subtitles, text, start, end)
    return 0


def _merge_overlapping_subtitles(subtitles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not subtitles:
        return []
    ordered = sorted(
        subtitles,
        key=lambda x: (
            float(x.get("start_time", 0.0) or 0.0),
            float(x.get("end_time", 0.0) or 0.0),
        ),
    )
    merged: List[Dict[str, Any]] = []
    for item in ordered:
        text = _clean_subtitle_text(item.get("msg") or "")
        if not _is_meaningful_subtitle_text(text):
            continue
        start = max(0.0, float(item.get("start_time", 0.0) or 0.0))
        end = max(start, float(item.get("end_time", start) or start))

        if merged:
            prev = merged[-1]
            prev_text = str(prev.get("msg") or "")
            prev_end = float(prev.get("end_time", 0.0) or 0.0)

            if text == prev_text and abs(start - float(prev.get("start_time", 0.0) or 0.0)) < 0.01:
                prev["end_time"] = max(prev_end, end)
                continue

            if start <= prev_end + 0.12 and (len(text) <= 4 or len(prev_text) <= 4):
                joined = _clean_subtitle_text(prev_text + text)
                if _is_meaningful_subtitle_text(joined):
                    prev["msg"] = joined
                    prev["end_time"] = max(prev_end, end)
                    continue

        merged.append({"msg": text, "start_time": start, "end_time": end})
    return merged


def _subtitles_to_srt(subtitles: List[Dict[str, Any]]) -> str:
    idx = 1
    lines: List[str] = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx,
                    text,
                    subtitle.get("start_time"),
                    subtitle.get("end_time"),
                )
            )
            idx += 1
    return "\n".join(lines) + ("\n" if lines else "")


def _write_text_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _build_segments_payload(
    subtitles: List[Dict[str, Any]],
    *,
    source: str,
    backend: str,
) -> List[Dict[str, Any]]:
    payload = []
    for idx, item in enumerate(subtitles, start=1):
        payload.append(
            {
                "id": idx,
                "start": float(item.get("start_time", 0.0) or 0.0),
                "end": float(item.get("end_time", 0.0) or 0.0),
                "text": str(item.get("msg") or ""),
                "source": source,
                "backend": backend,
                "confidence": None,
            }
        )
    return payload


def _write_segments_json(path: str, subtitles: List[Dict[str, Any]], *, source: str, backend: str):
    payload = _build_segments_payload(subtitles, source=source, backend=backend)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _create_with_sensevoice(audio_file: str, subtitle_file: str = ""):
    if not _load_sensevoice_model():
        return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    raw_srt_path, clean_srt_path = _derive_debug_paths(subtitle_file)
    segments_json_path = _derive_segments_json_path(subtitle_file)

    forced_language = _normalize_language_code(DEFAULT_FORCE_LANGUAGE) or "zh"
    logger.info(
        "开始 SenseVoice-Small 转写: "
        f"audio={audio_file}, size={_safe_file_size_mb(audio_file):.2f} MB, "
        f"language={forced_language}, batch_size_s={DEFAULT_FUNASR_BATCH_SIZE_S}, "
        f"merge_vad=True, merge_length_s={DEFAULT_FUNASR_MERGE_LENGTH_S}, use_itn={DEFAULT_FUNASR_USE_ITN}"
    )

    start = timer()
    try:
        results = model.generate(
            input=audio_file,
            cache={},
            language=forced_language,
            use_itn=DEFAULT_FUNASR_USE_ITN,
            batch_size_s=DEFAULT_FUNASR_BATCH_SIZE_S,
            merge_vad=True,
            merge_length_s=DEFAULT_FUNASR_MERGE_LENGTH_S,
            output_timestamp=True,
        )
    except TypeError:
        results = model.generate(
            input=audio_file,
            cache={},
            language=forced_language,
            use_itn=DEFAULT_FUNASR_USE_ITN,
            batch_size_s=DEFAULT_FUNASR_BATCH_SIZE_S,
            merge_vad=True,
            merge_length_s=DEFAULT_FUNASR_MERGE_LENGTH_S,
        )

    subtitles: List[Dict[str, Any]] = []
    raw_subtitles: List[Dict[str, Any]] = []
    raw_items = 0

    if isinstance(results, dict):
        results = [results]

    logger.info(f"SenseVoice raw result sample: {str(results)[:1200]}")

    for item in results or []:
        if not isinstance(item, dict):
            continue
        raw_items += 1
        _parse_funasr_result_item(item, subtitles, raw_subtitles)

    raw_subtitles = _merge_overlapping_subtitles(raw_subtitles)
    subtitles = _merge_overlapping_subtitles(subtitles)

    end = timer()
    logger.info(
        f"SenseVoice-Small complete, elapsed: {end - start:.2f} s, "
        f"raw_items={raw_items}, raw_lines={len(raw_subtitles)}, clean_lines={len(subtitles)}"
    )
    if not subtitles:
        logger.error("SenseVoice-Small 未生成有效字幕行")
        return None

    logger.warning(
        f"SUBTITLE_CLEAN_WRITE count={len(subtitles)} "
        f"first3={[x.get('msg') for x in subtitles[:3]]}"
    )

    raw_srt = _subtitles_to_srt(raw_subtitles)
    clean_srt = _subtitles_to_srt(subtitles)

    _write_text_file(raw_srt_path, raw_srt)
    _write_text_file(clean_srt_path, clean_srt)
    _write_text_file(subtitle_file, clean_srt)
    _write_segments_json(segments_json_path, subtitles, source="auto_clean", backend="sensevoice")

    logger.info(f"raw subtitle file created: {raw_srt_path}")
    logger.info(f"clean subtitle file created: {clean_srt_path}")
    logger.info(f"subtitle file created: {subtitle_file}")
    logger.info(f"segments json created: {segments_json_path}")

    return subtitle_file if os.path.exists(subtitle_file) else None


def _create_with_faster_whisper(audio_file: str, subtitle_file: str = ""):
    global model
    if not _load_faster_whisper_model():
        return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    raw_srt_path, clean_srt_path = _derive_debug_paths(subtitle_file)
    segments_json_path = _derive_segments_json_path(subtitle_file)

    forced_language = _normalize_language_code(DEFAULT_FORCE_LANGUAGE)
    transcribe_kwargs = dict(
        audio=audio_file,
        beam_size=max(1, DEFAULT_BEAM_SIZE),
        best_of=max(1, DEFAULT_BEST_OF),
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=DEFAULT_VAD_MIN_SILENCE_MS),
        condition_on_previous_text=True,
        temperature=0.0,
        initial_prompt=DEFAULT_INITIAL_PROMPT,
    )
    if forced_language:
        transcribe_kwargs["language"] = forced_language

    logger.info(
        "开始 faster-whisper 转写: "
        f"audio={audio_file}, size={_safe_file_size_mb(audio_file):.2f} MB, "
        f"language={forced_language or 'auto'}, beam_size={transcribe_kwargs['beam_size']}, "
        f"best_of={transcribe_kwargs['best_of']}, word_timestamps={transcribe_kwargs['word_timestamps']}, "
        f"vad_min_silence={DEFAULT_VAD_MIN_SILENCE_MS}ms"
    )

    segments, info = model.transcribe(**transcribe_kwargs)

    logger.info(f"检测到的语言: '{info.language}', probability: {info.language_probability:.2f}")

    start = timer()
    subtitles: List[Dict[str, Any]] = []
    raw_subtitles: List[Dict[str, Any]] = []
    segment_count = 0

    for segment in segments:
        segment_count += 1
        raw_text = getattr(segment, "text", "") or ""
        seg_start = float(getattr(segment, "start", 0.0) or 0.0)
        seg_end = float(getattr(segment, "end", 0.0) or 0.0)
        if not raw_text:
            continue
        _append_raw_subtitle_line(raw_subtitles, raw_text, seg_start, seg_end)
        _append_split_subtitle_lines(subtitles, raw_text, seg_start, seg_end)

    raw_subtitles = _merge_overlapping_subtitles(raw_subtitles)
    subtitles = _merge_overlapping_subtitles(subtitles)

    end = timer()
    logger.info(
        f"complete, elapsed: {end - start:.2f} s, raw_segments={segment_count}, "
        f"raw_lines={len(raw_subtitles)}, clean_lines={len(subtitles)}"
    )
    if not subtitles:
        logger.error("faster-whisper 未生成有效字幕行")
        return None

    raw_srt = _subtitles_to_srt(raw_subtitles)
    clean_srt = _subtitles_to_srt(subtitles)

    _write_text_file(raw_srt_path, raw_srt)
    _write_text_file(clean_srt_path, clean_srt)
    _write_text_file(subtitle_file, clean_srt)
    _write_segments_json(segments_json_path, subtitles, source="auto_clean", backend="faster-whisper")

    logger.info(f"raw subtitle file created: {raw_srt_path}")
    logger.info(f"clean subtitle file created: {clean_srt_path}")
    logger.info(f"subtitle file created: {subtitle_file}")
    logger.info(f"segments json created: {segments_json_path}")
    return subtitle_file if os.path.exists(subtitle_file) else None


def create(audio_file, subtitle_file: str = ""):
    backend = CURRENT_BACKEND
    if backend in {"sensevoice", "funasr"}:
        result = _create_with_sensevoice(audio_file, subtitle_file)
        if result:
            return result
        logger.warning("SenseVoice-Small 字幕生成失败，回退到 faster-whisper")
    return _create_with_faster_whisper(audio_file, subtitle_file)


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []
    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length) if max_length else 1.0


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    script_lines = utils.split_string_by_punctuations(video_script)
    return subtitle_items, script_lines


def create_with_gemini(audio_file: str, subtitle_file: str, api_key: str):
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = "生成这段语音的转录文本。请以SRT格式输出，包含时间戳。"
    try:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        response = gemini_model.generate_content([prompt, audio_data])
        transcript = response.text
        if not subtitle_file:
            subtitle_file = f"{audio_file}.srt"
        with open(subtitle_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        logger.info(f"Gemini生成的字幕文件已保存: {subtitle_file}")
        return subtitle_file
    except Exception as e:
        logger.error(f"使用Gemini处理音频时出错: {e}")
        return None


def extract_audio_and_create_subtitle(video_file: str, subtitle_file: str = "") -> Optional[str]:
    audio_file = ""
    video = None
    try:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        audio_dir = utils.temp_dir("audio_extract")
        os.makedirs(audio_dir, exist_ok=True)
        audio_file = os.path.join(audio_dir, f"{video_name}_speech.mp3")

        if not subtitle_file:
            subtitle_dir = utils.temp_dir("subtitles")
            os.makedirs(subtitle_dir, exist_ok=True)
            subtitle_file = os.path.join(subtitle_dir, f"{video_name}.srt")

        logger.info(f"开始从视频提取音频: {video_file}")

        extracted = False
        if _ffmpeg_available():
            extracted = _extract_audio_ffmpeg(video_file, audio_file)

        if not extracted:
            logger.info("回退到 moviepy 提取音频")
            video = VideoFileClip(video_file)
            if video.audio is None:
                logger.error("视频文件不包含音频轨道，无法自动生成字幕")
                return None

            audio_file = os.path.join(audio_dir, f"{video_name}_speech.wav")
            logger.info(f"正在提取音频到: {audio_file}")
            video.audio.write_audiofile(
                audio_file,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", str(DEFAULT_AUDIO_CHANNELS), "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE)],
                logger=None,
            )
            logger.info(f"moviepy 音频提取完成: {audio_file} ({_safe_file_size_mb(audio_file):.2f} MB)")

        logger.info("音频提取完成，开始生成字幕")
        result = create(audio_file, subtitle_file)
        if result and os.path.exists(result):
            logger.info(f"字幕生成完成: {result}")
            return result
        logger.error("字幕生成失败，未输出字幕文件")
        return None
    except Exception as e:
        logger.error(f"处理视频文件时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    finally:
        try:
            if video is not None:
                video.close()
        except Exception:
            pass
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                logger.info("已清理临时音频文件")
            except Exception as e:
                logger.warning(f"清理临时音频文件失败: {e}")
