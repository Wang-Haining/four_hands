import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


class ResultAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.load_results()

    def load_results(self):
        """Load all result files from the results directory."""
        for result_file in self.results_dir.glob("*.json"):
            with result_file.open('r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Extract key metrics
                    result = {
                        'model': data['config']['model'].split('/')[-1],
                        'stage': data['config']['stage'],
                        'temperature': data['config']['temperature'],
                        'test_doc': data['config']['test_doc'],
                        'predicted_author': data['aggregated_result']['author'],
                        'confidence': data['aggregated_result']['confidence'],
                        'vote_distribution': data['aggregated_result'][
                            'vote_distribution']
                    }
                    self.results_data.append(result)
                except Exception as e:
                    print(f"Error loading {result_file}: {str(e)}")

    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of results."""
        df = pd.DataFrame(self.results_data)

        # pivot table for better visualization
        summary = pd.pivot_table(
            df,
            values='confidence',
            index=['model', 'stage'],
            columns=['temperature'],
            aggfunc='mean'
        ).round(3)

        return summary

    def plot_heatmap(self, output_file: str = "heatmap.png"):
        """Generate heatmap visualization of results."""
        df = pd.DataFrame(self.results_data)

        # create pivot table for heatmap
        heatmap_data = pd.pivot_table(
            df,
            values='confidence',
            index=['model', 'stage'],
            columns=['temperature'],
            aggfunc='mean'
        )

        # set up the matplotlib figure
        plt.figure(figsize=(12, 8))

        # create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='YlOrRd',
            fmt='.3f',
            cbar_kws={'label': 'Confidence'}
        )

        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def plot_performance_comparison(self, output_file: str = "performance.png"):
        """Generate bar plot comparing model performances across stages."""
        df = pd.DataFrame(self.results_data)

        plt.figure(figsize=(15, 8))

        # create grouped bar plot
        sns.barplot(
            data=df,
            x='stage',
            y='confidence',
            hue='model',
            ci='sd'
        )

        plt.title('Model Performance Comparison Across Stages')
        plt.xlabel('Stage')
        plt.ylabel('Confidence')
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def generate_latex_table(self) -> str:
        """Generate LaTeX table of results."""
        summary = self.create_summary_table()

        # convert to LaTeX format
        latex_table = summary.to_latex(
            float_format="%.3f",
            caption="Model Performance Across Different Configurations",
            label="tab:results"
        )

        return latex_table

    def save_analysis(self, output_dir: str = "analysis_output"):
        """Save all analysis outputs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # save summary table
        summary = self.create_summary_table()
        summary.to_csv(output_path / "summary_table.csv")

        # save LaTeX table
        with (output_path / "latex_table.tex").open('w') as f:
            f.write(self.generate_latex_table())

        # generate plots
        self.plot_heatmap(str(output_path / "heatmap.png"))
        self.plot_performance_comparison(str(output_path / "performance.png"))

        # generate detailed report
        report = self.generate_report()
        with (output_path / "analysis_report.txt").open('w') as f:
            f.write(report)

    def generate_report(self) -> str:
        """Generate a detailed analysis report."""
        df = pd.DataFrame(self.results_data)

        report = []
        report.append("Authorship Analysis Grid Search Results")
        report.append("=" * 50)

        # overall best configuration
        best_config = df.loc[df['confidence'].idxmax()]
        report.append("\nBest Overall Configuration:")
        report.append(f"Model: {best_config['model']}")
        report.append(f"Stage: {best_config['stage']}")
        report.append(f"Temperature: {best_config['temperature']}")
        report.append(f"Confidence: {best_config['confidence']:.3f}")

        # performance by model
        report.append("\nPerformance by Model:")
        model_perf = df.groupby('model')['confidence'].agg(['mean', 'std']).round(3)
        report.append(model_perf.to_string())

        # performance by stage
        report.append("\nPerformance by Stage:")
        stage_perf = df.groupby('stage')['confidence'].agg(['mean', 'std']).round(3)
        report.append(stage_perf.to_string())

        return "\n".join(report)


def main():
    analyzer = ResultAnalyzer()
    analyzer.save_analysis()

    # print summary to console
    print("\nAnalysis Summary:")
    print("=" * 50)
    print("\nSummary Table:")
    print(analyzer.create_summary_table())
    print("\nAnalysis outputs have been saved to the 'analysis_output' directory.")


if __name__ == "__main__":
    main()
