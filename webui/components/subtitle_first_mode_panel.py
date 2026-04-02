from __future__ import annotations

import streamlit as st

MODE_LABELS = {
    "fast": "快速",
    "balanced": "标准",
    "quality": "高质量",
}

VISUAL_LABELS = {
    "off": "关闭",
    "auto": "自动",
    "boost": "强化",
}

STYLE_LABELS = {
    "general": "通用",
    "short_drama": "短剧",
    "documentary": "纪实",
}

STRATEGY_LABELS = {
    "chronological": "顺叙",
    "highlight_first": "先高能后回讲",
}

ACCURACY_LABELS = {
    "high": "准确优先",
    "balanced": "均衡",
}


def _safe_index(options, current, default):
    if current in options:
        return options.index(current)
    return options.index(default)


def render_subtitle_first_mode_panel(tr):
    with st.container(border=True):
        st.markdown("### " + tr("影视解说设置"))

        generation_options = ["fast", "balanced", "quality"]
        visual_options = ["off", "auto", "boost"]
        style_options = ["general", "short_drama", "documentary"]
        strategy_options = ["chronological", "highlight_first"]
        accuracy_options = ["high", "balanced"]

        st.selectbox(
            tr("生成模式"),
            options=generation_options,
            index=_safe_index(generation_options, st.session_state.get("generation_mode", "balanced"), "balanced"),
            format_func=lambda x: MODE_LABELS.get(x, x),
            help=tr("快速更省成本，标准更均衡，高质量会生成更细的剧情块与更多关键帧"),
            key="generation_mode",
        )

        st.selectbox(
            tr("目标时长"),
            options=[3, 5, 7, 8, 10, 20],
            index=[3, 5, 7, 8, 10, 20].index(st.session_state.get("target_duration_minutes", 8)) if st.session_state.get("target_duration_minutes", 8) in [3,5,7,8,10,20] else 3,
            key="target_duration_minutes",
            help=tr("软约束，最终成片允许比目标值浮动几十秒，优先保证讲准确"),
        )

        st.selectbox(
            tr("叙事策略"),
            options=strategy_options,
            index=_safe_index(strategy_options, st.session_state.get("narrative_strategy", "chronological"), "chronological"),
            format_func=lambda x: STRATEGY_LABELS.get(x, x),
            key="narrative_strategy",
        )

        st.selectbox(
            tr("准确性策略"),
            options=accuracy_options,
            index=_safe_index(accuracy_options, st.session_state.get("accuracy_priority", "high"), "high"),
            format_func=lambda x: ACCURACY_LABELS.get(x, x),
            help=tr("准确优先：宁可少讲一点，也不要讲错"),
            key="accuracy_priority",
        )

        st.selectbox(
            tr("视觉补充"),
            options=visual_options,
            index=_safe_index(visual_options, st.session_state.get("visual_mode", "auto"), "auto"),
            format_func=lambda x: VISUAL_LABELS.get(x, x),
            help=tr("关闭=纯字幕；自动=只给关键剧情块补帧；强化=重要块多补几帧"),
            key="visual_mode",
        )

        st.selectbox(
            tr("解说风格"),
            options=style_options,
            index=_safe_index(style_options, st.session_state.get("narration_style", "general"), "general"),
            format_func=lambda x: STYLE_LABELS.get(x, x),
            help=tr("影响脚本表达方式，不改变事实判断"),
            key="narration_style",
        )

        st.caption(tr("说明：影视解说主链已改为‘整剧理解 -> 剧情块 -> 回原片补帧 -> 分段写脚本’。"))
