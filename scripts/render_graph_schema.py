import os
from graphviz import Digraph

# Keep this in sync with compute_graph_emb_generic_npy.py
SCHEMA = [
    ("User", "Item", "RATED"),
    ("User", "Review", "WROTE"),
    ("Review", "Item", "DESCRIBES"),
    ("Review", "Sentiment", "HAS_POLARITY"),
    ("Item", "Popularity", "HAS_POP"),
    ("Review", "Review", "REVIEW_SIM"),
]

# Optional: show weights in edge labels (keep in sync with REL_WEIGHT)
REL_WEIGHT = {
    "RATED":        1.0,
    "WROTE":        1.2,
    "DESCRIBES":    0.8,
    "HAS_POLARITY": 0.7,
    "HAS_POP":      0.5,
    "REVIEW_SIM":   0.8,
}

NODE_STYLE = {
    "User":       dict(shape="oval",      color="#1f77b4"),
    "Review":     dict(shape="box",       color="#d62728"),
    "Item":       dict(shape="box",       color="#2ca02c"),
    "Sentiment":  dict(shape="diamond",   color="#9467bd"),
    "Popularity": dict(shape="diamond",   color="#8c564b"),
}

def render_schema(out_dir: str, name: str = "task3_graph_schema", show_weights: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    dot = Digraph(name=name, format="png")
    dot.attr(rankdir="LR", splines="spline", concentrate="true")
    dot.attr("node", style="rounded,filled", fillcolor="white", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", fontsize="10", color="#444444")

    # Add nodes (unique)
    nodes = set()
    for s, t, _ in SCHEMA:
        nodes.add(s); nodes.add(t)

    for n in sorted(nodes):
        st = NODE_STYLE.get(n, {})
        dot.node(n, n, **st)

    # Add edges with relation labels (and optional weights)
    for s, t, rel in SCHEMA:
        if show_weights and rel in REL_WEIGHT:
            label = f"{rel} (w={REL_WEIGHT[rel]})"
        else:
            label = rel
        dot.edge(s, t, label=label)

    png_path = dot.render(filename=name, directory=out_dir, cleanup=True)

    # Also emit SVG (better for report quality)
    dot.format = "svg"
    svg_path = dot.render(filename=name, directory=out_dir, cleanup=True)

    return png_path, svg_path

if __name__ == "__main__":
    out_png, out_svg = render_schema(
        out_dir="/Users/in22339881/gitrepo/uoa/mc/PPlug-uoa/graph_emb",
        name="task3_graph_schema",
        show_weights=True,
    )
    print("Wrote:")
    print(" -", out_png)
    print(" -", out_svg)