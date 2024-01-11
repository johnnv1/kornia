import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path

pols = {
    "regular": [(0.45, 0.05), (0.93, 0.19), (0.95, 0.7), (0.47, 0.87), (0.168, 0.47)],
    "irregular": [(0.15, 0.47), (0.45, 0.23), (0.79, 0.35), (0.95, 0.7), (0.55, 0.7)],
    "concave": [(0.18, 0.32), (0.79, 0.17), (0.56, 0.48), (0.88, 0.77), (0.44, 0.82)],
    "complex": [(0.24, 0.18), (0.85, 0.23), (0.08, 0.64), (0.89, 0.67), (0.40, 0.91)],
    "with interiors": [
        [(0.20, 0.20), (0.80, 0.20), (0.50, 0.70)],
        [[(0.40, 0.25), (0.40, 0.30), (0.45, 0.30), (0.45, 0.25)], [(0.50, 0.35), (0.50, 0.40), (0.55, 0.40)]],
    ],
    "multipolygon": [
        [(0.09, 0.01), (0.18, 0.04), (0.2, 0.14), (0.1, 0.17), (0.032, 0.1)],
        [(0.18, 0.32), (0.79, 0.17), (0.56, 0.48), (0.88, 0.77), (0.44, 0.82)],
    ],
}


def plot_points(points, ax):
    xs, ys = zip(*points)
    ax.fill(xs, ys, color="orange", linewidth=3)

    ax.scatter(xs, ys, color="red")
    for i, j in points:
        ax.text(i, j + 0.05, f"({i}, {j})")


fig, axes = plt.subplots(2, 3, dpi=150, subplot_kw={"aspect": "equal"})

for (name, points), ax in zip(pols.items(), [s for i in axes for s in i]):
    if len(points) == 0:
        continue
    elif isinstance(points[0], tuple):
        plot_points(points, ax)
    elif isinstance(points[-1][0], tuple):
        for p in points:
            plot_points(p, ax)

    else:
        xs, ys = zip(*points[0])
        path = Path.make_compound_path(Path(points[0]), *[Path(ring) for ring in points[1]])
        patch = PathPatch(path, facecolor="green")
        collection = PatchCollection([patch], facecolor="green")
        ax.add_collection(collection)

    ax.set_ylim(1, 0)
    ax.set_xlim(0, 1)

    ax.title.set_text(name)

plt.show()
