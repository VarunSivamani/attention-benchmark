"""
Generate HTML comparison report from training metrics of both models.

Reads metrics.jsonl from both runs and generates interactive visualizations.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_metrics(jsonl_path: str) -> List[Dict]:
    """Load metrics from JSONL file."""
    metrics = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                metrics.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {jsonl_path} not found")
    return metrics


def generate_html_report(
    flash_metrics_file: str = "runs/flash_gqa/metrics.jsonl",
    nsa_metrics_file: str = "runs/nsa_gqa/metrics.jsonl",
    output_file: str = "comparison_report.html",
) -> None:
    """Generate comparison HTML report."""
    flash_metrics = load_metrics(flash_metrics_file)
    nsa_metrics = load_metrics(nsa_metrics_file)

    if not flash_metrics and not nsa_metrics:
        print("No metrics found. Run training first.")
        return

    html_content = _build_html(flash_metrics, nsa_metrics)

    Path(output_file).write_text(html_content)
    print(f"Report saved to {output_file}")


def _build_html(flash_metrics: List[Dict], nsa_metrics: List[Dict]) -> str:
    """Build HTML with embedded charts."""
    flash_steps = [m.get("step", 0) for m in flash_metrics]
    flash_losses = [m.get("loss", 0) for m in flash_metrics]
    flash_ppls = [m.get("ppl", 0) for m in flash_metrics]
    flash_tps = [m.get("tokens_per_sec", 0) for m in flash_metrics]

    nsa_losses = [m.get("loss", 0) for m in nsa_metrics]
    nsa_ppls = [m.get("ppl", 0) for m in nsa_metrics]
    nsa_tps = [m.get("tokens_per_sec", 0) for m in nsa_metrics]

    flash_final_loss = flash_losses[-1] if flash_losses else 0
    flash_final_ppl = flash_ppls[-1] if flash_ppls else 0
    flash_avg_tps = np.mean(flash_tps) if flash_tps else 0

    nsa_final_loss = nsa_losses[-1] if nsa_losses else 0
    nsa_final_ppl = nsa_ppls[-1] if nsa_ppls else 0
    nsa_avg_tps = np.mean(nsa_tps) if nsa_tps else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Attention Comparison: Flash vs NSA</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
        }}
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            padding: 2rem;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            border-left: 4px solid #667eea;
        }}
        .metric-card.nsa {{
            border-left-color: #764ba2;
        }}
        .metric-card h3 {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}
        .metric-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }}
        .metric-card .unit {{
            font-size: 0.9rem;
            color: #999;
            margin-left: 0.5rem;
        }}
        .chart-section {{
            padding: 2rem;
            border-top: 1px solid #eee;
        }}
        .chart-section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #333;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin-bottom: 2rem;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 1.5rem 2rem;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Attention Mechanism Comparison</h1>
            <p>Flash Attention + GQA vs Native Sparse Attention + GQA on 124M GPT-style Model</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Flash Final Loss</h3>
                <div class="value">{flash_final_loss:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>NSA Final Loss</h3>
                <div class="value">{nsa_final_loss:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Flash Final Perplexity</h3>
                <div class="value">{flash_final_ppl:.2f}</div>
            </div>
            <div class="metric-card nsa">
                <h3>NSA Final Perplexity</h3>
                <div class="value">{nsa_final_ppl:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Flash Avg Throughput</h3>
                <div class="value">{flash_avg_tps:.0f}<span class="unit">tok/s</span></div>
            </div>
            <div class="metric-card nsa">
                <h3>NSA Avg Throughput</h3>
                <div class="value">{nsa_avg_tps:.0f}<span class="unit">tok/s</span></div>
            </div>
        </div>

        <div class="chart-section">
            <h2>Training Loss Convergence</h2>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <div class="chart-section">
            <h2>Tokens per Second (Throughput)</h2>
            <div class="chart-container">
                <canvas id="tpsChart"></canvas>
            </div>
        </div>

        <div class="chart-section">
            <h2>Perplexity</h2>
            <div class="chart-container">
                <canvas id="pplChart"></canvas>
            </div>
        </div>

        <div class="footer">
            <p>Generated from training metrics. See runs/*/metrics.jsonl for raw data.</p>
        </div>
    </div>

    <script>
        const ctx1 = document.getElementById('lossChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'line',
            data: {{
                labels: {json.dumps(flash_steps)},
                datasets: [
                    {{
                        label: 'Flash Attention',
                        data: {json.dumps(flash_losses)},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                    {{
                        label: 'Native Sparse Attention',
                        data: {json.dumps(nsa_losses)},
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }},
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Loss' }} }},
                    x: {{ title: {{ display: true, text: 'Training Step' }} }},
                }}
            }}
        }});

        const ctx2 = document.getElementById('tpsChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'line',
            data: {{
                labels: {json.dumps(flash_steps)},
                datasets: [
                    {{
                        label: 'Flash Attention',
                        data: {json.dumps(flash_tps)},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                    {{
                        label: 'Native Sparse Attention',
                        data: {json.dumps(nsa_tps)},
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }},
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Tokens/Second' }} }},
                    x: {{ title: {{ display: true, text: 'Training Step' }} }},
                }}
            }}
        }});

        const ctx3 = document.getElementById('pplChart').getContext('2d');
        new Chart(ctx3, {{
            type: 'line',
            data: {{
                labels: {json.dumps(flash_steps)},
                datasets: [
                    {{
                        label: 'Flash Attention',
                        data: {json.dumps(flash_ppls)},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                    {{
                        label: 'Native Sparse Attention',
                        data: {json.dumps(nsa_ppls)},
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.4,
                        fill: true,
                    }},
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }},
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Perplexity' }} }},
                    x: {{ title: {{ display: true, text: 'Training Step' }} }},
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-metrics", default="runs/flash_gqa/metrics.jsonl")
    parser.add_argument("--nsa-metrics", default="runs/nsa_gqa/metrics.jsonl")
    parser.add_argument("--output", default="comparison_report.html")
    args = parser.parse_args()

    generate_html_report(
        flash_metrics_file=args.flash_metrics,
        nsa_metrics_file=args.nsa_metrics,
        output_file=args.output,
    )
