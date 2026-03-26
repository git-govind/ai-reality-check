"""
app.py — Navigation entry-point for VeritasIQ.

Defines sidebar page order and labels using st.navigation / st.Page.
Run with:  streamlit run app.py
"""
import sys
import os
import base64
import json
import pathlib

# Ensure the project root is on sys.path before any page or module imports run.
# This is required so that packages like `utils/`, `models/`, `profiler.py`, etc.
# are resolvable regardless of how or from where Streamlit is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components

pg = st.navigation([
    st.Page("pages/1_Dashboard.py",       title="Dashboard",       icon="📊"),
    st.Page("pages/2_Text_Evaluator.py",  title="Prompt Evaluator", icon="🔍"),
    st.Page("pages/3_Image_Evaluator.py", title="Image Evaluator",  icon="🖼️"),
    st.Page("pages/4_Voice_Evaluator.py", title="Voice Evaluator", icon="🎙️"),
    st.Page("pages/5_Video_Evaluator.py", title="Video Evaluator", icon="🎥"),
])

# ── Sidebar logo (st.logo places content above nav links) ────────────────────
# Full logo: name + tagline — displayed when sidebar is expanded
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 70">'
    '<defs>'
    '<linearGradient id="viq-grad" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#3b82f6"/>'
    '<stop offset="100%" stop-color="#a855f7"/>'
    '</linearGradient>'
    '<linearGradient id="viq-icon-grad" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#3b82f6"/>'
    '<stop offset="100%" stop-color="#a855f7"/>'
    '</linearGradient>'
    '</defs>'
    '<text x="0" y="44"'
    ' font-family="ui-sans-serif,system-ui,-apple-system,Arial,sans-serif"'
    ' font-size="42" font-weight="900" letter-spacing="-1"'
    ' fill="url(#viq-grad)">VeritasIQ</text>'
    '<text x="0" y="63"'
    ' font-family="ui-sans-serif,system-ui,-apple-system,Arial,sans-serif"'
    ' font-size="14" fill="#94a3b8" font-style="italic">'
    'Where Intelligence Meets Integrity</text>'
    '</svg>'
)
# Icon: shown in the collapsed sidebar
_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    '<defs>'
    '<linearGradient id="vi-grad" x1="0%" y1="0%" x2="100%" y2="0%">'
    '<stop offset="0%" stop-color="#3b82f6"/>'
    '<stop offset="100%" stop-color="#a855f7"/>'
    '</linearGradient>'
    '</defs>'
    '<text x="2" y="30"'
    ' font-family="ui-sans-serif,system-ui,-apple-system,Arial,sans-serif"'
    ' font-size="26" font-weight="900" fill="url(#vi-grad)">V</text>'
    '</svg>'
)

def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

try:
    st.logo(image=_svg_uri(_LOGO_SVG), icon_image=_svg_uri(_ICON_SVG), size="large")
except TypeError:
    # size parameter not available in this Streamlit version
    st.logo(image=_svg_uri(_LOGO_SVG), icon_image=_svg_uri(_ICON_SVG))

_STATIC = pathlib.Path(__file__).parent / "static"

def _js_str(path: pathlib.Path) -> str:
    """JSON-encode file content and escape </ so it can't close a <script> tag."""
    return json.dumps(path.read_text(encoding="utf-8")).replace("</", "<\\/")

_arch_js = _js_str(_STATIC / "architecture.html")
_pipe_js = _js_str(_STATIC / "pipeline_guide.html")

components.html(f"""
<script>
var _docs = [
    {{ icon: '🏗\uFE0F', label: 'Architecture Guide', html: {_arch_js} }},
    {{ icon: '🔬', label: 'Pipeline Guide',      html: {_pipe_js} }}
];

function injectSidebar() {{
    var doc = window.parent.document;

    // ── Coming soon badges ──────────────────────────────────────────────
    var links = doc.querySelectorAll('[data-testid="stSidebarNavLink"]');
    links.forEach(function(link, i) {{
        if ((i === 3 || i === 4) && !link.querySelector('.cs-badge')) {{
            var badge = doc.createElement('span');
            badge.className = 'cs-badge';
            badge.textContent = 'Coming soon...';
            badge.style.cssText = 'display:block;color:#9ca3af;font-style:italic;font-size:0.78em;margin-top:1px;pointer-events:none;';
            link.appendChild(badge);
        }}
    }});

    // ── Docs links ──────────────────────────────────────────────────────
    var nav = doc.querySelector('[data-testid="stSidebarNav"]');
    if (!nav || nav.querySelector('.viq-docs-section')) return;

    var section = doc.createElement('div');
    section.className = 'viq-docs-section';
    section.style.cssText = 'padding:0.5rem 0 0 0;margin-top:0.25rem;border-top:1px solid rgba(148,163,184,0.2);';

    _docs.forEach(function(d) {{
        var html = d.html;
        var a = doc.createElement('a');
        a.href = 'javascript:void(0)';
        a.style.cursor = 'pointer';
        a.addEventListener('click', function(e) {{
            e.preventDefault();
            var w = window.parent.open('', '_blank');
            if (w) {{
                w.document.open();
                w.document.write(html);
                w.document.close();
            }}
        }});
        a.style.cssText = [
            'display:flex', 'align-items:center', 'gap:0.5rem',
            'padding:0.4rem 1rem', 'border-radius:0.375rem',
            'text-decoration:none', 'color:inherit', 'cursor:pointer',
            'font-size:0.875rem', 'transition:background 0.15s'
        ].join(';');
        a.onmouseover = function() {{ this.style.background = 'rgba(148,163,184,0.1)'; }};
        a.onmouseout  = function() {{ this.style.background = 'transparent'; }};

        var icon = doc.createElement('span');
        icon.textContent = d.icon;

        var label = doc.createElement('span');
        label.textContent = d.label;

        a.appendChild(icon);
        a.appendChild(label);
        section.appendChild(a);
    }});

    nav.appendChild(section);
}}

injectSidebar();
setTimeout(injectSidebar, 300);
setTimeout(injectSidebar, 1000);
</script>
""", height=0)

pg.run()

