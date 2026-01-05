import csv
import sys

def create_svg_line_chart(output_file):
    width = 600
    height = 400
    padding = 50

    zenith_loss = []
    conv_loss = []
    steps = []

    try:
        with open('benchmark_results_accuracy.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['Step']))
                zenith_loss.append(float(row['ZenithLoss']))
                conv_loss.append(float(row['ConvLoss']))
    except FileNotFoundError:
        print("CSV not found")
        return

    if not steps:
        return

    max_loss = max(max(zenith_loss), max(conv_loss))
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

    # Grid lines (Y)
    for i in range(5):
        y_val = max_loss * i / 4.0
        y = height - padding - y_val * y_scale
        svg += f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="#ddd" stroke-dasharray="4" />\n'
        svg += f'<text x="{padding-5}" y="{y+5}" font-family="Arial" font-size="10" text-anchor="end">{y_val:.2f}</text>\n'

    # Plot Zenith (Blue)
    points_z = []
    for s, l in zip(steps, zenith_loss):
        x = padding + s * x_scale
        y = height - padding - l * y_scale
        points_z.append(f"{x},{y}")

    svg += f'<polyline points="{" ".join(points_z)}" fill="none" stroke="#3498db" stroke-width="2" />\n'

    # Plot Conv (Red)
    points_c = []
    for s, l in zip(steps, conv_loss):
        x = padding + s * x_scale
        y = height - padding - l * y_scale
        points_c.append(f"{x},{y}")

    svg += f'<polyline points="{" ".join(points_c)}" fill="none" stroke="#e74c3c" stroke-width="2" />\n'

    # Legend
    svg += f'<rect x="{width - 150}" y="20" width="15" height="15" fill="#3498db" />\n'
    svg += f'<text x="{width - 130}" y="32" font-family="Arial" font-size="12">Zenith AE</text>\n'
    svg += f'<rect x="{width - 150}" y="40" width="15" height="15" fill="#e74c3c" />\n'
    svg += f'<text x="{width - 130}" y="52" font-family="Arial" font-size="12">Conv Baseline</text>\n'

    # Title
    svg += f'<text x="{width/2}" y="25" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold">Training Loss (20 Steps)</text>\n'

    svg += '</svg>'

    with open(output_file, 'w') as f:
        f.write(svg)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    create_svg_line_chart('docs/accuracy_benchmark.svg')
