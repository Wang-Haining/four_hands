"""
Authorship analysis for '哀弦篇' using LLMs.

This module implements a voting-based approach to determine whether passages from
'哀弦篇' were written by Lu Xun, Zhou Zuoren, or collaboratively. It supports
multiple analysis strategies (basic/zero-shot/few-shot/chain-of-thought) and
various LLM backends (local or remote).
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from collections import defaultdict

import openai
from pydantic import BaseModel, Field
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from vllm import LLM, SamplingParams

from utils import load_corpus


def rearrange_train_val_while_reserve_shots(train: list, val: list, test: list) -> \
tuple[list, list, list, dict, dict]:
    """
    Rearrange train and val sets while reserving first sample of each author from training for few-shot.
    The remaining samples will be used to create a balanced validation set.

    Args:
        train: Original training documents
        val: Original validation documents
        test: Original test documents (unchanged)

    Returns:
        Tuple of (new_train, new_val, test, lx_shot, zzr_shot) where shots are individual documents
    """
    # get first sample of each author from training
    lx_shot = next(doc for doc in train if doc['author'] == 'lx')
    zzr_shot = next(doc for doc in train if doc['author'] == 'zzr')
    new_train = [lx_shot, zzr_shot]

    # combine remaining training and validation documents
    remaining_train = [doc for doc in train if doc not in new_train]
    all_docs = remaining_train + val

    # create balanced validation set using 15 docs from each author
    lx_docs = [doc for doc in all_docs if doc['author'] == 'lx'][:15]
    zzr_docs = [doc for doc in all_docs if doc['author'] == 'zzr'][:15]
    new_val = lx_docs + zzr_docs

    print("\nNew Split Summary:")
    print(
        f"Training (Few-shot): {len(new_train)} docs ({sum(1 for d in new_train if d['author'] == 'lx')} LX, {sum(1 for d in new_train if d['author'] == 'zzr')} ZZR)")
    print(
        f"Validation: {len(new_val)} docs ({sum(1 for d in new_val if d['author'] == 'lx')} LX, {sum(1 for d in new_val if d['author'] == 'zzr')} ZZR)")
    print(f"Test: {len(test)} docs")

    return new_train, new_val, test, lx_shot, zzr_shot


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Evaluate predictions against ground truth."""
    # map full names to lowercase short labels for predictions
    name_to_code = {
        "Lu Xun": "lx",
        "Zhou Zuoren": "zzr"
    }

    # extract predictions and true labels
    y_true = [doc['author'] for doc in ground_truth]
    y_pred = [name_to_code[pred['aggregated_result']['author']] for pred in predictions]

    # calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                               average='weighted')

    # calculate confusion matrix with lowercase labels
    labels = ["lx", "zzr"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # format confusion matrix as dict for json serialization
    conf_matrix_dict = {
        'matrix': conf_matrix.tolist(),
        'labels': labels
    }

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': conf_matrix_dict,
        'support': {label: sum(1 for y in y_true if y == label) for label in labels}
    }


class ExperimentLogger:
    """Handle logging of experimental results."""

    def __init__(self):
        self.output_dir = Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_id(self, config: Dict) -> str:
        """Generate unique experiment identifier."""
        model_short = config['model'].split('/')[-1]
        return (f"{model_short}"
                f"_stage-{config['stage']}"
                f"_temp-{config['temperature']}")

    def save_experiment(self, config: Dict, val_docs: List[Dict],
                        test_docs: List[Dict], prompt_dict: Dict[str, str],
                        val_results: List[Dict], test_results: List[Dict]):
        """Save experiment results including validation and test predictions."""
        experiment_id = self.get_experiment_id(config)

        # evaluate on validation set
        val_metrics = evaluate_predictions(val_results, val_docs)

        # organize results for each test document
        test_predictions = []
        for doc, result in zip(test_docs, test_results):
            test_predictions.append({
                'title': doc['title'],
                'prediction': result
            })

        # organize experiment data
        experiment_data = {
            "config": config,
            "prompt": prompt_dict,
            "validation": {
                "metrics": val_metrics,
                "predictions": val_results
            },
            "test_predictions": test_predictions
        }

        # save results
        output_file = self.output_dir / f"{experiment_id}.json"
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        print(f"\nExperiment results saved to: {output_file}")
        self._print_experiment_summary(experiment_data)

    def _print_experiment_summary(self, data: Dict):
        """Print a summary of the experiment results."""
        # print basic experiment info
        print("\nExperiment Summary:")
        print(f"Model: {data['config']['model']}")
        print(f"Stage: {data['config']['stage']}")
        print(f"Temperature: {data['config']['temperature']}")

        # print validation metrics
        print("\nValidation Metrics:")
        metrics = data['validation']['metrics']
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")

        # print class distribution
        print("\nClass Distribution:")
        for label, count in metrics['support'].items():
            print(f"{label}: {count}")

        # print explanation summary if available
        if any('explanation' in result for result in data['validation']['predictions']):
            print("\nExplanation Summary:")
            # collect importance scores across all documents
            all_importance = defaultdict(list)
            for result in data['validation']['predictions']:
                if 'explanation' in result:
                    for feature, importance in result['explanation'][
                        'feature_importance'].items():
                        all_importance[feature].append(importance)

            # calculate and show average importance scores
            print("\nAverage Feature Importance (top 10):")
            avg_importance = {
                feature: np.mean(scores)
                for feature, scores in all_importance.items()
            }
            # sort by absolute importance and show top 10
            for feature, importance in sorted(
                    avg_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
            )[:10]:
                print(f"{feature}: {importance:.3f}")


class AuthorshipResult(BaseModel):
    """Schema for the authorship analysis result."""
    author: Literal['Lu Xun', 'Zhou Zuoren'] = Field(
        description="The predicted author: must be either 'Lu Xun' or 'Zhou Zuoren'"
    )
    analysis: Optional[str] = Field(
        default=None,
        description="Optional analysis explaining the reasoning behind the attribution"
    )


class PromptManager:
    """Enhanced prompt manager with simpler prediction format."""

    def __init__(self, use_cot: bool = False):
        self.use_cot = use_cot
        self.train = None
        self.val = None
        self.test = None
        self.lx_shot = None
        self.zzr_shot = None

        self.system_msg = """You are an expert in Chinese literature, specializing in stylometric analysis. Your task is to analyze passages from the disputed work 哀弦篇 to determine their authorship between Lu Xun and Zhou Zuoren.

        IMPORTANT: Base your analysis STRICTLY on the stylistic markers provided. Do NOT use any authorship knowledge you may have about other works by Lu Xun or Zhou Zuoren, as this could lead to unreliable conclusions. Focus solely on analyzing the presence and patterns of the specific markers in the given text.

        You should analyze the text and make the prediction at the end with your prediction wrapped in {PREDICTION_START} and {PREDICTION_END} tags. The prediction must be exactly one of these two authors, with no additional text within the prediction tags."""

    def setup_data(self, train: list, val: list, test: list):
        """Setup data splits and shot samples."""
        self.train, self.val, self.test, self.lx_shot, self.zzr_shot = rearrange_train_val_while_reserve_shots(
            train, val, test)

    def _get_basic(self) -> str:
        """Return basic analysis instructions."""
        return """Analyze the writing styles of the input texts, disregarding differences in topic and content, and reason based on linguistic features such as character frequency."""

    def _get_knowledge(self) -> str:
        """Return linguistic knowledge for zero-shot prompting."""
        return """Features supporting Lu Xun's style include: 之, 而, 唯, 矣, 是, 于是, 足以, 必, 何, 徒, 然, 不, 乃, 于, 则, 进而, 全, 光, 夫. Features supporting Zhou Zuoren's style include: 本, 及, 别, 原, 各, 为, 多, 但, 自然, 随."""

    def _get_examples(self) -> str:
        """Generate examples using the shot samples."""
        if not (self.lx_shot and self.zzr_shot):
            raise ValueError("Shot samples not initialized. Call setup_data first.")

        examples = []

        # format Lu Xun example
        examples.append(
            f"Text:\n{self.lx_shot['text']}\n\n"
            f"Analysis: ...\n"
            f"{{PREDICTION_START}}Lu Xun{{PREDICTION_END}}"
        )

        # format Zhou Zuoren example
        examples.append(
            f"Text:\n{self.zzr_shot['text']}\n\n"
            f"Analysis: ...\n"
            f"{{PREDICTION_START}}Zhou Zuoren{{PREDICTION_END}}"
        )

        return "\n\n".join(examples)

    def construct_prompt(self, text: str, stage: str) -> Dict[str, str]:
        """Construct the full prompt with new prediction format.

        Args:
            text: The text to analyze
            stage: Analysis stage ("basic", "zero-shot", "few-shot", "cot")
        """
        if stage in ["few-shot", "cot"] and not (self.lx_shot and self.zzr_shot):
            raise ValueError(
                f"Shot samples required for {stage} stage. Call setup_data first.")

        prompt_parts = [self._get_basic()]

        if stage in ["zero-shot", "few-shot", "cot"]:
            prompt_parts.append("Linguistic Features to Consider:")
            prompt_parts.append(self._get_knowledge())

        if stage in ["few-shot", "cot"]:
            prompt_parts.append("Reference Examples:")
            prompt_parts.append(self._get_examples())

        prompt_parts.extend([
            "Text to Analyze:",
            text,
            "",
            "Provide your analysis and end with your prediction wrapped in {PREDICTION_START} and {PREDICTION_END} tags.",
        ])

        return {
            "system": self.system_msg,
            "user": "\n\n".join(prompt_parts)
        }


class ModelManager:
    """Modified model manager to handle new prediction format."""

    def __init__(self, model_name: str, temperature: float = 0.6, seed: int = 42):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.is_openai = model_name.startswith("gpt-")

        if not self.is_openai:
            self.model = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=2 if "70B" in model_name else 1,
                enforce_eager=False,
                quantization="awq" if "70B" in model_name else None,
                device="cuda",
            )

    def _format_messages(self, system_msg: str, user_msg: str) -> List[Dict[str, str]]:
        """Format chat messages."""
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

    def _extract_prediction(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract prediction and analysis from model output."""
        try:
            # find the prediction between tags
            start_tag = "{PREDICTION_START}"
            end_tag = "{PREDICTION_END}"
            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag)

            if start_idx == -1 or end_idx == -1:
                return None

            prediction = text[start_idx + len(start_tag):end_idx].strip()
            analysis = text[:start_idx].strip() if start_idx > 0 else ""

            # validate prediction
            if prediction not in ["Lu Xun", "Zhou Zuoren"]:
                return None

            return {
                "author": prediction,
                "analysis": analysis
            }

        except Exception as e:
            print(f"Prediction extraction error: {str(e)}")
            return None

    def generate(self, prompt: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Generate a single response with new prediction format."""
        messages = self._format_messages(prompt["system"], prompt["user"])

        try:
            if self.is_openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )
                result = response.choices[0].message.content
            else:
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=8192
                )

                result = self.model.chat(
                    messages=messages,
                    sampling_params=sampling_params
                )
                result = result[0].outputs[0].text

            return self._extract_prediction(result)

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None

    def generate_with_retries(self, prompt: Dict[str, str], num_runs: int,
                              max_retries: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple responses with retry logic and robust error handling."""
        valid_results = []
        attempts = 0

        while len(valid_results) < num_runs and attempts < num_runs * max_retries:
            try:
                if self.is_openai:
                    openai.api_key = os.environ["OPENAI_API_KEY"]

                result = self.generate(prompt)
                if result is not None and isinstance(result,
                                                     dict) and 'author' in result:
                    valid_results.append(result)

                attempts += 1

            except Exception as e:
                print(f"Error during generation attempt {attempts}: {str(e)}")
                attempts += 1
                continue

        success_rate = len(valid_results) / attempts if attempts > 0 else 0
        print(f"\nGeneration statistics:")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Total attempts: {attempts}")
        print(f"Valid results: {len(valid_results)}")

        # always return a list, even if empty
        return valid_results


def create_example_json(text: str, author: str, analysis: str = "") -> str:
    """Create a valid JSON example for few-shot learning."""
    example = AuthorshipResult(author=author, analysis=analysis)
    return json.dumps(example.model_dump(), ensure_ascii=False, indent=2)


def compute_vote_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated results from multiple runs.
    Assumes all results are valid (no ERROR states).
    """
    possible_authors = ["Lu Xun", "Zhou Zuoren"]

    # count votes
    vote_counts = Counter(result["author"] for result in results)
    total_votes = len(results)

    # calculate probabilities for each class
    pred_proba = {author: vote_counts.get(author, 0) / total_votes
                 for author in possible_authors}

    # get winner and confidence
    winner = max(pred_proba.items(), key=lambda x: x[1])[0]
    confidence = pred_proba[winner]

    # aggregate analysis from majority class
    majority_analyses = [r.get("analysis", "") for r in results if r["author"] == winner]
    analysis = random.choice(majority_analyses) if majority_analyses else "No analysis available"

    return {
        "author": winner,
        "confidence": confidence,
        "analysis": analysis,
        "pred_proba": pred_proba,
        "vote_distribution": pred_proba.copy()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced authorship analysis using LLMs")
    # existing arguments
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name (Hugging Face or OpenAI)")
    parser.add_argument("--stage", choices=["basic", "zero-shot", "few-shot", "cot"],
                        required=True, help="Analysis stage to run")
    parser.add_argument("--num_runs", type=int, default=30,
                        help="Number of runs for voting")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    # add explanation arguments
    parser.add_argument("--explain", action="store_true",
                        help="Generate LIME-like explanations")
    parser.add_argument("--explain_samples", type=int, default=1000,
                        help="Number of samples for explanation (if --explain)")
    parser.add_argument("--runs_per_sample", type=int, default=3,
                        help="Number of runs per perturbed sample (if --explain)")
    args = parser.parse_args()

    # init core components
    random.seed(args.seed)
    train, val, test = load_corpus()
    model_mgr = ModelManager(args.model, args.temperature, args.seed)
    prompt_mgr = PromptManager(use_cot=(args.stage == "cot"))
    logger = ExperimentLogger()

    # setup data splits
    prompt_mgr.setup_data(train, val, test)

    # init explainer if needed
    explainer = None
    if args.explain:
        from explainer import CharacterExplainer
        explainer = CharacterExplainer(
            n_samples=args.explain_samples,
            runs_per_sample=args.runs_per_sample
        )

    # process validation set
    print("\nRunning validation set evaluation...")
    val_results = []
    for val_doc in prompt_mgr.val:  # use val from prompt_mgr
        print(f"\nValidating on document: {val_doc['title']}")
        prompt = prompt_mgr.construct_prompt(
            val_doc["text"],
            stage=args.stage
        )
        run_results = model_mgr.generate_with_retries(prompt, args.num_runs)

        # prepare base result dictionary
        result_dict = {
            'doc': val_doc,
            'results': run_results,
            'aggregated_result': compute_vote_results(run_results)
        }

        # add explanation if requested
        if explainer:
            print("Generating explanation...")
            explanation = explainer.explain_prediction(
                text=val_doc["text"],
                model_manager=model_mgr,
                prompt_manager=prompt_mgr,
                stage=args.stage
            )
            result_dict['explanation'] = explanation

        val_results.append(result_dict)

    # process test set with same logic
    print("\nRunning test set prediction...")
    test_results = []
    for test_doc in prompt_mgr.test:  # use test from prompt_mgr
        print(f"\nAnalyzing document: {test_doc['title']}")
        prompt = prompt_mgr.construct_prompt(
            test_doc["text"],
            stage=args.stage
        )
        run_results = model_mgr.generate_with_retries(prompt, args.num_runs)

        result_dict = {
            'doc': test_doc,
            'results': run_results,
            'aggregated_result': compute_vote_results(run_results)
        }

        if explainer:
            print("Generating explanation...")
            explanation = explainer.explain_prediction(
                text=test_doc["text"],
                model_manager=model_mgr,
                prompt_manager=prompt_mgr,
                stage=args.stage
            )
            result_dict['explanation'] = explanation

        test_results.append(result_dict)

    # save all results
    config = vars(args).copy()
    logger.save_experiment(config, prompt_mgr.val, prompt_mgr.test, prompt, val_results,
                           test_results)


if __name__ == "__main__":
    main()
