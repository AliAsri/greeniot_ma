"""Reusable UI blocks for the Streamlit dashboard."""

import streamlit as st


def render_section_card(label: str, title: str, copy: str, style: str = "") -> None:
    st.markdown(
        f"""
<div class="section-card" style="margin-bottom: 1rem; {style}">
  <div class="section-label">{label}</div>
  <div class="section-title">{title}</div>
  <div class="section-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_status_card(label: str, title: str, copy: str, detail: str, detail_color: str, background: str) -> None:
    st.markdown(
        f"""
<div class="section-card" style="margin-bottom: 1rem; background: {background};">
  <div class="section-label">{label}</div>
  <div class="section-title">{title}</div>
  <div class="section-copy">{copy}</div>
  <div class="section-copy" style="margin-top:0.45rem; font-weight:600; color:{detail_color};">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_takeaway_card(label: str, title: str, copy: str, accent: str = "#166a4a") -> None:
    st.markdown(
        f"""
<div class="section-card" style="margin-bottom: 1rem; border-left: 6px solid {accent}; background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,244,236,0.92));">
  <div class="section-label">{label}</div>
  <div class="section-title">{title}</div>
  <div class="section-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )
