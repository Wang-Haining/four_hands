import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import seaborn as sns


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
                    # extract config and validation metrics
                    result = {
                        'model': data['config']['model'].split('/')[-1],
                        'stage': data['config']['stage'],
                        'temperature': data['config']['temperature'],
                        'accuracy': data['validation_results']['metrics']['accuracy'],
                        'precision': data['validation_results']['metrics']['precision'],
                        'recall': data['validation_results']['metrics']['recall'],
                        'f1': data['validation_results']['metrics']['f1'],
                        'confusion_matrix': data['validation_results']['metrics'][
                            'confusion_matrix'],
                        'test_predictions': data['test_results']['document_predictions']
                    }
                    self.results_data.append(result)
                except Exception as e:
                    print(f"Error loading {result_file}: {str(e)}")

    def create_summary_table(self) -> Dict[str, pd.DataFrame]:
        """Create a summary table of validation results."""
        df = pd.DataFrame(self.results_data)

        # create pivot tables for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        summaries = {}

        for metric in metrics:
            summaries[metric] = pd.pivot_table(
                df,
                values=metric,
                index=['model', 'stage'],
                columns=['temperature'],
                aggfunc='mean'
            ).round(3)

        return summaries

    def plot_heatmap(self, output_dir: Path):
        """Generate heatmap visualizations for each metric."""
        df = pd.DataFrame(self.results_data)
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        for metric in metrics:
            # create pivot table for heatmap
            heatmap_data = pd.pivot_table(
                df,
                values=metric,
                index=['model', 'stage'],
                columns=['temperature'],
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                cmap='YlOrRd',
                fmt='.3f',
                cbar_kws={'label': metric.capitalize()}
            )

            plt.title(f'Model Performance Heatmap - {metric.capitalize()}')
            plt.tight_layout()
            plt.savefig(output_dir / f"heatmap_{metric}.png")
            plt.close()

    def plot_confusion_matrices(self, output_dir: Path):
        """Generate confusion matrix plots for best configurations."""
        df = pd.DataFrame(self.results_data)

        # find best configuration based on F1 score
        best_config = df.loc[df['f1'].idxmax()]

        # plot confusion matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(best_config['confusion_matrix'])

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['LX', 'ZZR'],
            yticklabels=['LX', 'ZZR']
        )

        plt.title(f'Confusion Matrix for Best Configuration\n' +
                  f"({best_config['model']}, {best_config['stage']}, " +
                  f"temp={best_config['temperature']})")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / "best_confusion_matrix.png")
        plt.close()

    def create_test_predictions_table(self) -> pd.DataFrame:
        """Create a table of test set predictions for the best configuration."""
        df = pd.DataFrame(self.results_data)
        best_config = df.loc[df['f1'].idxmax()]

        # extract test predictions from best configuration
        test_preds = best_config['test_predictions']

        # convert to DataFrame
        test_df = pd.DataFrame([
            {
                'document': doc,
                'predicted_author': data['predicted_author'],
                'confidence': data['confidence'],
                'lx_votes': data['vote_distribution']['LX'],
                'zzr_votes': data['vote_distribution']['ZZR']
            }
            for doc, data in test_preds.items()
        ])

        return test_df

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for all metrics."""
        summaries = self.create_summary_table()
        latex_tables = {}

        for metric, summary in summaries.items():
            latex_tables[metric] = summary.to_latex(
                float_format="%.3f",
                caption=f"Model Performance ({metric.capitalize()}) Across Configurations",
                label=f"tab:results_{metric}"
            )

        return latex_tables

    def save_analysis(self, output_dir: str = "analysis_output"):
        """Save all analysis outputs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # save summary tables
        summaries = self.create_summary_table()
        for metric, summary in summaries.items():
            summary.to_csv(output_path / f"summary_{metric}.csv")

        # save test predictions
        test_preds = self.create_test_predictions_table()
        test_preds.to_csv(output_path / "test_predictions.csv", index=False)

        # save LaTeX tables
        latex_tables = self.generate_latex_tables()
        for metric, latex in latex_tables.items():
            with (output_path / f"latex_table_{metric}.tex").open('w') as f:
                f.write(latex)

        # generate plots
        self.plot_heatmap(output_path)
        self.plot_confusion_matrices(output_path)

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

        # best configuration based on F1 score
        best_config = df.loc[df['f1'].idxmax()]
        report.append("\nBest Overall Configuration (based on F1 score):")
        report.append(f"Model: {best_config['model']}")
        report.append(f"Stage: {best_config['stage']}")
        report.append(f"Temperature: {best_config['temperature']}")
        report.append(f"F1 Score: {best_config['f1']:.3f}")
        report.append(f"Accuracy: {best_config['accuracy']:.3f}")
        report.append(f"Precision: {best_config['precision']:.3f}")
        report.append(f"Recall: {best_config['recall']:.3f}")

        # performance by model
        report.append("\nPerformance by Model (F1 Score):")
        model_perf = df.groupby('model')['f1'].agg(['mean', 'std']).round(3)
        report.append(model_perf.to_string())

        # performance by stage
        report.append("\nPerformance by Stage (F1 Score):")
        stage_perf = df.groupby('stage')['f1'].agg(['mean', 'std']).round(3)
        report.append(stage_perf.to_string())

        # test set predictions for best configuration
        test_preds = self.create_test_predictions_table()
        report.append("\nTest Set Predictions (Best Configuration):")
        report.append(test_preds.to_string())

        return "\n".join(report)


def main():
    analyzer = ResultAnalyzer()
    analyzer.save_analysis()

    # print summary to console
    print("\nAnalysis Summary:")
    print("=" * 50)
    print("\nSummary Tables:")
    summaries = analyzer.create_summary_table()
    for metric, summary in summaries.items():
        print(f"\n{metric.upper()} Summary:")
        print(summary)
    print("\nAnalysis outputs have been saved to the 'analysis_output' directory.")


if __name__ == "__main__":
    main()
