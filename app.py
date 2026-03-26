"""
app.py — Navigation entry-point for VeritasIQ.

Defines sidebar page order and labels using st.navigation / st.Page.
Run with:  streamlit run app.py
"""
import sys
import os
import base64

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

components.html("""
<script>
function injectComingSoon() {
    var doc = window.parent.document;
    var links = doc.querySelectorAll('[data-testid="stSidebarNavLink"]');
    links.forEach(function(link, i) {
        if ((i === 3 || i === 4) && !link.querySelector('.cs-badge')) {
            var badge = doc.createElement('span');
            badge.className = 'cs-badge';
            badge.textContent = 'Coming soon...';
            badge.style.cssText = 'display:block;color:#9ca3af;font-style:italic;font-size:0.78em;margin-top:1px;pointer-events:none;';
            link.appendChild(badge);
        }
    });
}
injectComingSoon();
setTimeout(injectComingSoon, 300);
setTimeout(injectComingSoon, 1000);
</script>
""", height=0)

pg.run()

