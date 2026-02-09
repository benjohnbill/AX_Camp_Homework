"""
icons.py
Lucide Icons Subset for Antigravity
"""

def get_icon_svg(name, size=24, color="currentColor"):
    """Get SVG string for Lucide icon"""
    
    # Common attrs
    attrs = f'xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    
    icons = {
        "waves": '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>',
        "orbit": '<circle cx="12" cy="12" r="3"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><path d="M10.4 9.6 16.6 3.4"/><path d="M17.43 6.6a9.98 9.98 0 0 0-4.04-3.19"/><path d="M3.4 16.6 9.6 10.4"/><path d="M6.59 17.43a9.98 9.98 0 0 0 3.19 4.04"/>',
        "pen-tool": '<path d="m12 19 7-7 3 3-7 7-3-3z"/><path d="m18 13-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="m2 2 7.586 7.586"/><circle cx="11" cy="11" r="2"/>',
        "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "bandage": '<path d="M10 10.873 13.127 14"/><path d="m18.707 9.293-1.414-1.414a1 1 0 0 0-1.414 0l-1.63 1.63a1 1 0 0 0 0 1.414l1.414 1.414a1 1 0 0 0 1.414 0l1.63-1.63a1 1 0 0 0 0-1.414Z"/><path d="m10.293 17.707-1.414-1.414a1 1 0 0 0-1.414 0l-1.63 1.63a1 1 0 0 0 0 1.414l1.414 1.414a1 1 0 0 0 1.414 0l1.63-1.63a1 1 0 0 0 0-1.414Z"/><path d="m17.5 19.5-13-13"/>', # Actually using 'activity' might be better, but Bandage exists in Lucide as 'Bandage' (replaced by 'Plaster' or custom). Let's use 'Activity' here or custom. I will use 'Activity' for simplicity as Apology icon.
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M9 3v4"/><path d="M3 5h4"/><path d="M3 9h4"/>',
        "telescope": '<path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44"/><path d="m13.56 11.747 4.332-.924"/><path d="m16 21-3.105-6.21"/><path d="M16.485 5.94a2 2 0 0 1 2.618-1.872l2.374.65a2 2 0 0 1 1.461 2.453l-.538 2.15a2 2 0 0 1-2.618 1.873l-2.375-.65a2 2 0 0 1-1.46-2.453l.538-2.15Z"/><circle cx="12" cy="12" r="2"/>',
        "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
        "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.1.2-2.2.5-3.3.3-1.12.5-2.2.5-3.3a6.34 6.34 0 0 0-5 5.5c-.13 2.13.78 3.82 2.5 4.6Z"/>',
        "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        "shield-alert": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "book-open": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
        "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
        "minus": '<path d="M5 12h14"/>',
        "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>',
        "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
        "droplet": '<path d="M12 22a7 7 0 0 0 7-7c0-2-2-5-7-13-5 8-7 11-7 13a7 7 0 0 0 7 7Z"/>',
        "calendar": '<rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/>',
        "check": '<polyline points="20 6 9 17 4 12"/>',
        "key": '<circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/>',
        "pencil": '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>',
        "galaxy": '<path d="M21 12a9 9 0 1 1-9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>', # Custom approximate
        "meteor": '<path d="m13 2-9 9"/><path d="m11 5 3 2.5"/><path d="m7.5 8 2.5 3"/><path d="M6 13a6 6 0 1 0 12 0 6 6 0 1 0-12 0Z"/>' # Custom approximate logic removed, using generic trace
    }
    
    path = icons.get(name, "")
    if not path:
        return ""
        
    return f'<svg {attrs}>{path}</svg>'

def get_icon(name, size=20, color="currentColor"):
    """Get centralized icon HTML wrapper"""
    svg = get_icon_svg(name, size, color)
    return f'<span style="display:inline-block;vertical-align:middle;margin-right:8px;">{svg}</span>'
