import csv
import sys

def create_svg_line_chart(output_file):
    width = 800
    height = 500
    padding = 60

    steps = []
    loss_he = []
    loss_id = []
    loss_pe = []

    try:
        with open('benchmark_results_ablation.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['Step']))
                loss_he.append(float(row['Baseline']))
                loss_id.append(float(row['IdentityInit']))
                loss_pe.append(float(row['IdentityPE']))
    except FileNotFoundError:
        print("CSV not found")
        return

    if not steps:
        return

    max_loss = max(max(loss_he), max(loss_id), max(loss_pe))
    if max_loss == 0: max_loss = 1

    max_step = max(steps)

    # Scale
    x_scale = (width - 2 * padding) / max_step
    y_scale = (height - 2 * padding) / max_loss

    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg += f'<rect width="100%" height="100%" fill="white" />\n'

    # Axes
    svg += f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black" />\n'
    svg += f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="black" />\n'

    # Labels
    svg += f'<text x="{width/2}" y="{height-20}" font-family="Arial" font-size="14" text-anchor="middle">Training Steps</text>\n'
    svg += f'<text x="20" y="{height/2}" font-family="Arial" font-size="14" text-anchor="middle" transform="rotate(-90 20,{height/2})">MSE Loss</text>\n'

    # Grid lines (Y)
    for i in range(5):
        y_val = max_loss * i / 4.0
        y = height - padding - y_val * y_scale
        svg += f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="#ddd" stroke-dasharray="4" />\n'
        svg += f'<text x="{padding-5}" y="{y+5}" font-family="Arial" font-size="10" text-anchor="end">{y_val:.2f}</text>\n'

    def plot_line(data, color):
        points = []
        for s, l in zip(steps, data):
            x = padding + s * x_scale
            y = height - padding - l * y_scale
            points.append(f"{x},{y}")
        return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" />\n'

    svg += plot_line(loss_he, "#e74c3c")   # Red
    svg += plot_line(loss_id, "#f1c40f")   # Yellow
    svg += plot_line(loss_pe, "#3498db")   # Blue

    # Legend
    legend_x = width - 200
    legend_y = 50

    def add_legend_item(y, color, text):
        return f'<rect x="{legend_x}" y="{y}" width="15" height="15" fill="{color}" />\n' \
               f'<text x="{legend_x + 20}" y="{y+12}" font-family="Arial" font-size="12">{text}</text>\n'

    svg += add_legend_item(legend_y, "#e74c3c", "Baseline (He Init, No PE)")
    svg += add_legend_item(legend_y + 20, "#f1c40f", "Identity Init, No PE")
    svg += add_legend_item(legend_y + 40, "#3498db", "Identity Init + Pos Emb")

    # Title
    svg += f'<text x="{width/2}" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold">Zenith Ablation Study (2000 Steps)</text>\n'

    svg += '</svg>'

    with open(output_file, 'w') as f:
        f.write(svg)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    create_svg_line_chart('docs/ablation_benchmark.svg')
