#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import annotations

import json
import os
import re
import time
import traceback

import streamlit as st
from loguru import logger

from app.config import config
from app.services.subtitle_first_pipeline import run_subtitle_first_pipeline
from app.services.subtitle_text import read_subtitle_text

# 确保提供商被注册
import app.services.llm  # noqa: F401


def parse_and_fix_json(json_string):
    if not json_string or not json_string.strip():
        return None
    json_string = json_string.strip()
    try:
        return json.loads(json_string)
    except Exception:
        pass
    try:
        fixed_braces = json_string.replace("{{", "{").replace("}}", "}")
        return json.loads(fixed_braces)
    except Exception:
        pass
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", json_string, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
    except Exception:
        pass
    return None


def generate_script_short_sunmmary(params, subtitle_path, video_theme, temperature):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        progress_bar.progress(min(int(progress), 100))
        status_text.text(f"{progress}% - {message}" if message else f"进度: {progress}%")

    try:
        with st.spinner("正在生成脚本..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            subtitle_mode = st.session_state.get("subtitle_source_mode", "existing_subtitle")
            allow_auto_subtitle = subtitle_mode == "auto_subtitle"
            if not allow_auto_subtitle and (not subtitle_path or not os.path.exists(subtitle_path)):
                st.error("字幕文件不存在")
                return

            text_provider = config.app.get("text_llm_provider", "gemini").lower()
            text_api_key = config.app.get(f"text_{text_provider}_api_key", "")
            text_model = config.app.get(f"text_{text_provider}_model_name", "")
            text_base_url = config.app.get(f"text_{text_provider}_base_url", "")

            generation_mode = st.session_state.get("generation_mode", "balanced")
            visual_mode = st.session_state.get("visual_mode", "auto")
            narration_style = st.session_state.get("narration_style", "general")
            target_duration_minutes = st.session_state.get("target_duration_minutes", 8)
            narrative_strategy = st.session_state.get("narrative_strategy", "chronological")
            accuracy_priority = st.session_state.get("accuracy_priority", "high")

            actual_subtitle_path = subtitle_path or ""
            logger.info(
                "开始影视解说主链: mode={}, visual_mode={}, style={}, target_minutes={}, strategy={}, subtitle_mode={}, subtitle_path={}",
                generation_mode,
                visual_mode,
                narration_style,
                target_duration_minutes,
                narrative_strategy,
                subtitle_mode,
                "AUTO" if not actual_subtitle_path else actual_subtitle_path,
            )

            pipeline_result = run_subtitle_first_pipeline(
                video_path=params.video_origin_path,
                subtitle_path=actual_subtitle_path,
                text_api_key=text_api_key,
                text_base_url=text_base_url,
                text_model=text_model,
                style=narration_style,
                generation_mode=generation_mode,
                visual_mode=visual_mode,
                scene_overrides={
                    "target_duration_minutes": target_duration_minutes,
                    "narrative_strategy": narrative_strategy,
                    "accuracy_priority": accuracy_priority,
                    "video_title": video_theme,
                    "short_name": video_theme,
                    "temperature": temperature,
                },
                progress_callback=update_progress,
            )

            if pipeline_result.get("success") and pipeline_result.get("script_items"):
                logger.success("影视解说主链成功")
                st.session_state["video_clip_json"] = pipeline_result["script_items"]
                st.session_state["subtitle_first_evidence"] = pipeline_result.get("evidence", [])
                st.session_state["subtitle_first_global_summary"] = pipeline_result.get("global_summary", {})
                st.session_state["video_clip_json_path"] = pipeline_result.get(
                    "script_path", st.session_state.get("video_clip_json_path", "")
                )
                st.session_state["movie_story_plot_chunks"] = pipeline_result.get("plot_chunks", [])
                st.session_state["movie_story_frame_records"] = pipeline_result.get("frame_records", [])

                actual_generated_subtitle = pipeline_result.get("generated_saved_subtitle_path") or pipeline_result.get("subtitle_path", "")
                if actual_generated_subtitle and os.path.exists(actual_generated_subtitle):
                    st.session_state["subtitle_path"] = actual_generated_subtitle
                    st.session_state["last_generated_subtitle_path"] = actual_generated_subtitle
                    try:
                        subtitle_obj = read_subtitle_text(actual_generated_subtitle)
                        st.session_state["subtitle_content"] = subtitle_obj.text if subtitle_obj else ""
                    except Exception:
                        pass

                update_progress(100, "脚本生成完成！")
                st.success(
                    f"影视解说脚本生成成功！剧情块 {len(pipeline_result.get('plot_chunks', []))} 个，"
                    f"脚本片段 {len(pipeline_result.get('script_items', []))} 个。"
                )
                return

            logger.warning("影视解说主链失败: {}", pipeline_result.get("error", "unknown"))
            if allow_auto_subtitle:
                st.error("自动生成字幕失败，请检查后端字幕流水线日志")
                return

            st.error("影视解说主链失败，请检查日志")
            return

    except Exception as err:
        st.error(f"生成过程中发生错误: {str(err)}")
        logger.exception(f"影视解说主链异常\n{traceback.format_exc()}")
    finally:
        time.sleep(0.8)
