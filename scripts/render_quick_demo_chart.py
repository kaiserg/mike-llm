"""Render the archived quick-demo loss history as a dependency-free SVG."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def points(rows, key, left, top, width, height, maximum):
    last_step = rows[-1]["step"]
    return " ".join(
        f"{left + width * row['step'] / last_step:.1f},{top + height * (1 - row[key] / maximum):.1f}"
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/quick-demo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/images/quick-demo-loss.svg")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = report["history"]
    if len(rows) < 2 or rows[0]["step"] != 0 or rows[-1]["step"] != report["steps"]:
        raise ValueError("unexpected quick-demo history")
    left, top, width, height = 72, 28, 660, 300
    maximum = max(row[key] for row in rows for key in ("train_loss", "validation_loss")) * 1.08
    train = points(rows, "train_loss", left, top, width, height, maximum)
    validation = points(rows, "validation_loss", left, top, width, height, maximum)
    grid = []
    for index in range(5):
        y = top + height * index / 4
        value = maximum * (1 - index / 4)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        grid.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    ticks = []
    for row in rows:
        x = left + width * row["step"] / rows[-1]["step"]
        ticks.append(f'<line x1="{x:.1f}" y1="{top + height}" x2="{x:.1f}" y2="{top + height + 5}" class="axis"/>')
        ticks.append(f'<text x="{x:.1f}" y="{top + height + 22}" text-anchor="middle">{row["step"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="410" viewBox="0 0 800 410" role="img" aria-labelledby="title desc">
  <title id="title">Quick demo training and validation loss</title>
  <desc id="desc">Both losses fall from about 2.75 before training to about 0.3 after 150 updates.</desc>
  <style>
    text {{ font: 13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; fill: #24292f; }}
    .title {{ font-size: 20px; font-weight: 600; }}
    .grid {{ stroke: #d8dee4; stroke-width: 1; }}
    .axis {{ stroke: #57606a; stroke-width: 1.5; }}
    .train {{ fill: none; stroke: #0969da; stroke-width: 3; }}
    .validation {{ fill: none; stroke: #cf222e; stroke-width: 3; }}
  </style>
  <rect width="800" height="410" fill="#ffffff"/>
  <text x="400" y="22" text-anchor="middle" class="title">Learning a synthetic color sequence from random weights</text>
  {''.join(grid)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>
  {''.join(ticks)}
  <polyline points="{train}" class="train"/>
  <polyline points="{validation}" class="validation"/>
  <line x1="545" y1="373" x2="575" y2="373" class="train"/><text x="583" y="378">training loss</text>
  <line x1="665" y1="373" x2="695" y2="373" class="validation"/><text x="703" y="378">validation loss</text>
  <text x="{left + width / 2}" y="397" text-anchor="middle">training updates</text>
  <text x="18" y="{top + height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + height / 2})">cross-entropy loss</text>
</svg>
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(args.output)


if __name__ == "__main__":
    main()
