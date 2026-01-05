"""
Paper-Quality Visualization Module
Generates publication-ready figures for ICWS submission.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from src.models import Composition, CompositionProblem
from src.evaluation import MetricsCalculator, StatisticalAnalyzer


class PaperVisualizer:
    """Generates publication-quality visualizations."""
    
    # ICWS/ACM conference figure settings
    FIG_WIDTH = 3.5  # Single column width
    FIG_HEIGHT = 2.5
    DPI = 300
    FONT_SIZE = 10
    LINE_WIDTH = 1.5
    MARKER_SIZE = 4
    
    COLORS = {
        'NSGA-II-Green': '#2E86AB',
        'MOPSO-Green': '#A23B72',
        'QoS-Greedy': '#F18F01',
        'Energy-Greedy': '#C73E1D',
        'Random': '#6B705C',
        'GA-QoS': '#3A5A40'
    }
    
    MARKERS = {
        'NSGA-II-Green': 'o',
        'MOPSO-Green': 's',
        'QoS-Greedy': '^',
        'Energy-Greedy': 'D',
        'Random': 'v',
        'GA-QoS': 'p'
    }
    
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure matplotlib for publication."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': self.FONT_SIZE,
            'axes.labelsize': self.FONT_SIZE,
            'axes.titlesize': self.FONT_SIZE,
            'legend.fontsize': self.FONT_SIZE - 1,
            'xtick.labelsize': self.FONT_SIZE - 1,
            'ytick.labelsize': self.FONT_SIZE - 1,
            'figure.figsize': (self.FIG_WIDTH, self.FIG_HEIGHT),
            'figure.dpi': self.DPI,
            'savefig.dpi': self.DPI,
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.8,
            'lines.linewidth': self.LINE_WIDTH,
            'lines.markersize': self.MARKER_SIZE,
        })
    
    def plot_pareto_front(
        self,
        solutions_dict: Dict[str, List[Composition]],
        title: str = "Pareto Front Comparison",
        xlabel: str = "Response Time (ms)",
        ylabel: str = "Energy Consumption (J)",
        filename: str = "pareto_front.png"
    ):
        """Plot Pareto fronts for multiple algorithms."""
        fig, ax = plt.subplots()
        
        for algo_name, solutions in solutions_dict.items():
            if not solutions:
                continue
            
            objectives = np.array([s.objectives for s in solutions])
            objectives = objectives[np.argsort(objectives[:, 0])]
            
            color = self.COLORS.get(algo_name, '#333333')
            marker = self.MARKERS.get(algo_name, 'o')
            
            ax.plot(objectives[:, 0], objectives[:, 1],
                   marker=marker, linestyle='-', color=color,
                   label=algo_name, markersize=4, alpha=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_convergence(
        self,
        history_dict: Dict[str, List[float]],
        title: str = "Convergence Analysis",
        ylabel: str = "Hypervolume",
        filename: str = "convergence.png"
    ):
        """Plot convergence curves."""
        fig, ax = plt.subplots()
        
        for algo_name, history in history_dict.items():
            color = self.COLORS.get(algo_name, '#333333')
            ax.plot(history, color=color, label=algo_name, linewidth=1.5)
        
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_boxplot_comparison(
        self,
        metrics_data: Dict[str, Dict[str, List[float]]],
        metric: str = "hypervolume",
        title: str = "Algorithm Comparison",
        filename: str = "boxplot.png"
    ):
        """Create boxplot comparison of metrics."""
        fig, ax = plt.subplots()
        
        algo_names = list(metrics_data.keys())
        data = [metrics_data[algo][metric] for algo in algo_names]
        
        bp = ax.boxplot(data, labels=algo_names, patch_artist=True)
        
        for patch, algo_name in zip(bp['boxes'], algo_names):
            patch.set_facecolor(self.COLORS.get(algo_name, '#333333'))
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_tradeoff_curve(
        self,
        solutions: List[Composition],
        baseline_energy: float,
        title: str = "Energy-QoS Trade-off",
        filename: str = "tradeoff.png"
    ):
        """Plot energy savings vs QoS degradation."""
        fig, ax = plt.subplots()
        
        objectives = np.array([s.objectives for s in solutions])
        
        # Calculate energy savings and response time increase
        base_response = objectives[:, 0].min()
        base_energy = objectives[:, 1].min()
        
        response_increase = (objectives[:, 0] - base_response) / base_response * 100
        energy_savings = (baseline_energy - objectives[:, 1]) / baseline_energy * 100
        
        # Color by carbon emission
        carbon = objectives[:, 1] * 0.3  # Approximate carbon factor
        
        scatter = ax.scatter(response_increase, energy_savings, c=carbon,
                           cmap='RdYlGn', alpha=0.7, s=50)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Response Time Increase (%)")
        ax.set_ylabel("Energy Savings (%)")
        ax.set_title(title)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Carbon Emission (kg CO2)")
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_scalability(
        self,
        runtime_data: Dict[str, Dict[int, float]],
        title: str = "Scalability Analysis",
        filename: str = "scalability.png"
    ):
        """Plot runtime vs problem size."""
        fig, ax = plt.subplots()
        
        sizes = sorted(list(runtime_data['NSGA-II-Green'].keys()))
        
        for algo_name, data in runtime_data.items():
            color = self.COLORS.get(algo_name, '#333333')
            times = [data.get(s, 0) for s in sizes]
            ax.plot(sizes, times, marker='o', label=algo_name, color=color)
        
        ax.set_xlabel("Number of Service Categories")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_radar_chart(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Multi-Metric Comparison",
        filename: str = "radar.png"
    ):
        """Create radar chart for multi-metric comparison."""
        if metrics is None:
            metrics = ['hypervolume', 'spread', 'num_solutions', 'avg_energy']
        
        fig, ax = plt.subplots(figsize=(self.FIG_WIDTH * 1.2, self.FIG_HEIGHT * 1.2),
                              subplot_kw=dict(polar=True))
        
        algo_names = list(metrics_data.keys())
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        for algo_name in algo_names:
            values = [metrics_data[algo_name].get(m, 0) for m in metrics]
            # Normalize
            max_vals = [max(metrics_data[a].get(m, 1) for a in algo_names) for m in metrics]
            values_norm = [v / (m + 1e-10) for v, m in zip(values, max_vals)]
            values_norm += values_norm[:1]
            
            color = self.COLORS.get(algo_name, '#333333')
            ax.plot(angles, values_norm, 'o-', linewidth=1.5, label=algo_name, color=color)
            ax.fill(angles, values_norm, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], size=8)
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close()
        
        return str(filepath)


class ResultsFormatter:
    """Formats results for direct insertion into paper."""
    
    def format_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        columns: List[str] = None,
        caption: str = "",
        label: str = "tab:results"
    ) -> str:
        """Generate LaTeX table from results."""
        if columns is None:
            columns = ['hypervolume', 'spread', 'num_solutions', 'avg_energy']
        
        col_names = {
            'hypervolume': 'HV',
            'spread': 'Spread',
            'num_solutions': '|PF|',
            'avg_energy': 'Energy (J)',
            'min_energy': 'Min Energy',
            'avg_response_time': 'RT (ms)'
        }
        
        lines = []
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{l" + "r" * len(columns) + "}")
        lines.append("\\hline")
        lines.append("Algorithm" + " & " + " & ".join([col_names.get(c, c) for c in columns]) + " \\\\")
        lines.append("\\hline")
        
        for algo_name in sorted(results.keys()):
            row = [algo_name.replace('-', ' ')]
            for col in columns:
                val = results[algo_name].get(col, 0)
                if isinstance(val, float):
                    if col in ['hypervolume', 'spread', 'avg_energy', 'min_energy']:
                        row.append(f"{val:.2f}")
                    else:
                        row.append(f"{val:.0f}")
                else:
                    row.append(str(val))
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def format_comparison_text(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'hypervolume'
    ) -> str:
        """Generate comparison paragraph for paper."""
        for algo_name in sorted(results.keys(),
                                key=lambda x: results[x].get('hypervolume_mean', 0),
                                reverse=True):
        
        best = sorted_algos[0]
        second = sorted_algos[1] if len(sorted_algos) > 1 else None
        
        improvement = 0
        if second:
            improvement = ((results[best].get(metric, 0) - results[second].get(metric, 0)) 
                          / results[second].get(metric, 1) * 100)
        
        text = (
            f"Our proposed NSGA-II-Green achieves a hypervolume of "
            f"{results[best].get('hypervolume', 0):.2f}, representing a "
            f"{improvement:.1f}\\% improvement over the best baseline "
            f"({second}) with {results[second].get('hypervolume', 0):.2f}. "
            f"The algorithm found {results[best].get('num_solutions', 0)} "
            f"Pareto-optimal solutions with an average energy consumption of "
            f"{results[best].get('avg_energy', 0):.2f}J."
        )
        
        return text
    
    def format_statistical_results(
        self,
        comparisons: Dict[str, Dict],
        metric: str = 'hypervolume'
    ) -> str:
        """Format Wilcoxon test results."""
        lines = ["\\begin{itemize}"]
        
        for pair, result in comparisons.items():
            algo1, algo2 = pair
            if result['significant']:
                lines.append(f"\\item {algo1} significantly outperforms {algo2} "
                           f"(p = {result['p_value']:.4f})")
            else:
                lines.append(f"\\item No significant difference between {algo1} "
                           f"and {algo2} (p = {result['p_value']:.4f})")
        
        lines.append("\\end{itemize}")
        
        return "\n".join(lines)
    
    def save_json_results(
        self,
        results: Dict,
        filename: str = "experiment_results.json"
    ):
        """Save results as JSON for reproducibility."""
        output_dir = Path("results/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(filepath)
    
    def generate_results_summary(
        self,
        all_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict:
        """Generate summary statistics across multiple runs."""
        summary = {}
        
        for algo, runs in all_results.items():
            summary[algo] = {
                'hypervolume_mean': np.mean([r.get('hypervolume', 0) for r in runs.values()]),
                'hypervolume_std': np.std([r.get('hypervolume', 0) for r in runs.values()]),
                'energy_mean': np.mean([r.get('avg_energy', 0) for r in runs.values()]),
                'energy_std': np.std([r.get('avg_energy', 0) for r in runs.values()]),
                'num_solutions_mean': np.mean([r.get('num_solutions', 0) for r in runs.values()]),
                'runtime_mean': np.mean([r.get('runtime', 0) for r in runs.values()])
            }
        
        return summary
