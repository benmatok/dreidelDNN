import csv
import sys

def create_svg_bar_chart(data, output_file):
    # data: list of (label, forward, backward)
    width = 600
    height = 400
    padding = 50
    bar_width = 40
    group_gap = 100

    max_val = max(max(d[1], d[2]) for d in data)
    scale = (height - 2 * padding) / max_val

    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg += f'<rect width="100%" height="100%" fill="white" />\n'

    # Axes
    svg += f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black" />\n'
    svg += f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="black" />\n'

    # Bars
    x_start = padding + 50
    colors = ['#3498db', '#e74c3c'] # Blue (Fwd), Red (Bwd)

    for i, (label, fwd, bwd) in enumerate(data):
        x = x_start + i * (2 * bar_width + group_gap)

        # Forward
        h_fwd = fwd * scale
        y_fwd = height - padding - h_fwd
        svg += f'<rect x="{x}" y="{y_fwd}" width="{bar_width}" height="{h_fwd}" fill="{colors[0]}" />\n'
        svg += f'<text x="{x + bar_width/2}" y="{y_fwd - 5}" font-family="Arial" font-size="12" text-anchor="middle">{fwd:.3f}s</text>\n'

        # Backward
        h_bwd = bwd * scale
        y_bwd = height - padding - h_bwd
        svg += f'<rect x="{x + bar_width}" y="{y_bwd}" width="{bar_width}" height="{h_bwd}" fill="{colors[1]}" />\n'
        svg += f'<text x="{x + bar_width + bar_width/2}" y="{y_bwd - 5}" font-family="Arial" font-size="12" text-anchor="middle">{bwd:.3f}s</text>\n'

        # Label
        svg += f'<text x="{x + bar_width}" y="{height - padding + 20}" font-family="Arial" font-size="12" text-anchor="middle">{label}</text>\n'

    # Legend
    svg += f'<rect x="{width - 150}" y="20" width="15" height="15" fill="{colors[0]}" />\n'
    svg += f'<text x="{width - 130}" y="32" font-family="Arial" font-size="12">Forward Pass</text>\n'
    svg += f'<rect x="{width - 150}" y="40" width="15" height="15" fill="{colors[1]}" />\n'
    svg += f'<text x="{width - 130}" y="52" font-family="Arial" font-size="12">Backward Pass</text>\n'

    # Title
    svg += f'<text x="{width/2}" y="25" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold">Wavelet AE Benchmark (128x128)</text>\n'

    svg += '</svg>'

    with open(output_file, 'w') as f:
        f.write(svg)
    print(f"Graph saved to {output_file}")

def main():
    data = []
    try:
        with open('benchmark_results_wavelet.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append((row['Model'], float(row['Forward']), float(row['Backward'])))
    except FileNotFoundError:
        print("CSV file not found")
        sys.exit(1)

    create_svg_bar_chart(data, 'docs/wavelet_benchmark.svg')

if __name__ == "__main__":
    main()
