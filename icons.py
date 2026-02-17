"""
icons.py
Lucide Icons Subset for Antigravity - Optimized for robustness
"""

ICON_TEXT = {
    "waves": "🌊",
    "orbit": "🪐",
    "timer": "⏱️",
    "layout-dashboard": "📊",
    "book-open": "📖",
    "star": "⭐",
    "activity": "📈",
    "sparkles": "✨",
    "bell": "🔔",
    "shield-alert": "⚠️",
    "check-circle": "✅",
    "anchor": "⚓",
    "alert-octagon": "🛑",
    "skull": "💀",
    "flame": "🔥",
    "target": "🎯",
    "trash": "🗑️",
    "zap": "⚡",
    "plus-circle": "➕",
    "moon": "🌙",
    "calendar": "📅",
    "pencil": "✍️",
    "save": "💾",
}

def get_icon_svg(name, size=24, color="currentColor"):
    """Get SVG string for Lucide icon"""
    
    attrs = f'xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    
    icons = {
        "waves": '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>',
        "orbit": '<circle cx="12" cy="12" r="3"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><path d="M10.4 9.6 16.6 3.4"/><path d="M17.43 6.6a9.98 9.98 0 0 0-4.04-3.19"/><path d="M3.4 16.6 9.6 10.4"/><path d="M6.59 17.43a9.98 9.98 0 0 0 3.19 4.04"/>',
        "timer": '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="12" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>',
        "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
        "book-open": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
        "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M9 3v4"/><path d="M3 5h4"/><path d="M3 9h4"/>',
        "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
        "shield-alert": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
        "galaxy": '<path d="M21 12a9 9 0 1 1-9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
        "meteor": '<path d="m13 2-9 9"/><path d="m11 5 3 2.5"/><path d="m7.5 8 2.5 3"/><circle cx="12" cy="12" r="10"/>',
        "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "anchor": '<path d="M12 22V8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/><circle cx="12" cy="5" r="3"/>',
        "alert-octagon": '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
        "skull": '<path d="m12.5 17-.5-1-.5 1h1z"/><path d="M15 22a1 1 0 0 0 1-1v-1a2 2 0 0 0-2-2H10a2 2 0 0 0-2 2v1a1 1 0 0 0 1 1h6z"/><path d="M18 10a6 6 0 1 0-12 0c0 4.97 4.48 8.82 8.91 8.12A6 6 0 0 0 18 10z"/><circle cx="9" cy="10" r="1"/><circle cx="15" cy="10" r="1"/>',
        "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.1.2-2.2.5-3.3.3-1.12.5-2.2.5-3.3a6.34 6.34 0 0 0-5 5.5c-.13 2.13.78 3.82 2.5 4.6Z"/>',
        "brain": '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54Z"/>',
        "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
        "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H5c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "undo": '<path d="M3 7V5c0-1.1.9-2 2-2h2"/><path d="M3 17v2c0 1.1.9 2 2 2h2"/><path d="M17 3h2c1.1 0 2 .9 2 2v2"/><path d="M21 17v2c0 1.1-.9 2-2 2h2"/><path d="M9 13H5v-4"/><path d="M5 13c0-1.1.9-2 2-2h14"/>', # Corrected Lucide undo roughly
        "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
        "plus-circle": '<circle cx="12" cy="12" r="10"/><path d="M12 8v8"/><path d="M8 12h8"/>',
        "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'
    }
    
    path = icons.get(name, "")
    if not path: return ""
    return f'<svg {attrs}>{path}</svg>'

def get_icon(name, size=20, color="currentColor"):
    """Get centralized icon HTML wrapper"""
    svg = get_icon_svg(name, size, color)
    return f'<span style="display:inline-block;vertical-align:middle;margin-right:8px;">{svg}</span>'


def get_icon_text(name: str) -> str:
    """Plain-text fallback icon for Streamlit widgets that escape HTML."""
    return ICON_TEXT.get(name, "•")
