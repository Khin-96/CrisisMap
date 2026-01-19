import folium
from typing import List, Dict, Optional


def metric_card_html(title: str, value: str, delta: Optional[str] = None, gradient: Optional[str] = None) -> str:
    gradient_style = gradient if gradient is not None else "linear-gradient(135deg, #4ECDC4 0%, #56D0B5 100%)"
    delta_html = f"<div style='font-size:12px; opacity:.8'>{delta}</div>" if delta else ""
    html = (
        f"<div class='card' style='background: {gradient_style}; color: white; padding: 14px; border-radius: 12px;'>"
        f"<div class='card-title'>{title}</div>"
        f"<div class='card-value'>{value}</div>"
        f"{delta_html}"
        f"</div>"
    )
    return html


def story_panel_html(steps: List[Dict], index: int) -> str:
    if index < 0 or index >= len(steps):
        index = 0
    s = steps[index]
    html = (
        f"<div class='story-step'>"
        f"<h3 style='margin:0 0 6px 0'>{s.get('title','Story')}</h3>"
        f"<p style='margin:0'>{s.get('text','')}</p>"
        f"</div>"
    )
    return html


def render_hotspots_map(hotspots: List[Dict]):
    if not hotspots:
        return None
    avg_lat = sum(h.get('latitude', 0) for h in hotspots) / max(1, len(hotspots))
    avg_lon = sum(h.get('longitude', 0) for h in hotspots) / max(1, len(hotspots))
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)
    for hs in hotspots:
        folium.CircleMarker(
            location=[hs.get('latitude', 0), hs.get('longitude', 0)],
            radius=max(4, int(hs.get('event_count', 1)) // 2 if hs.get('event_count', 1) else 4),
            color='red', fill=True,
            popup=f"{hs.get('location','')}: {hs.get('event_count',0)} events"
        ).add_to(m)
    return m
