"""
Authorship analysis for '哀弦篇' using LLMs.

This module implements a voting-based approach to determine whether passages from
'哀弦篇' were written by Lu Xun, Zhou Zuoren, or collaboratively. It supports
multiple analysis strategies (basic/zero-shot/few-shot/chain-of-thought) and
various LLM backends (local or remote).
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional
import argparse
from vllm import LLM, SamplingParams
import openai
from utils import load_corpus


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Evaluate predictions against ground truth."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # extract predictions and true labels
    y_true = [doc['author'].upper() for doc in ground_truth]
    y_pred = [pred['aggregated_result']['author'] for pred in predictions]

    # calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                               average='weighted')

    # calculate confusion matrix
    labels = ["LX", "ZZR"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # format confusion matrix as dict for JSON serialization
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

        # save
        output_file = self.output_dir / f"{experiment_id}.json"
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        print(f"\nExperiment results saved to: {output_file}")
        self._print_experiment_summary(experiment_data)

    def _print_experiment_summary(self, data: Dict):
        """Print a summary of the experiment results."""
        print("\nExperiment Summary:")
        print(f"Model: {data['config']['model']}")
        print(f"Stage: {data['config']['stage']}")
        print(f"Temperature: {data['config']['temperature']}")

        print("\nValidation Metrics:")
        metrics = data['validation']['metrics']
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")

        print("\nClass Distribution:")
        for label, count in metrics['support'].items():
            print(f"{label}: {count}")


class PromptManager:
    """Manage different prompting strategies for authorship analysis."""

    def __init__(self, use_cot: bool = False):
        self.use_cot = use_cot
        # system prompt
        self.system_msg = """You are an expert in Chinese literature, specializing in 
stylometric analysis. Your task is to analyze passages from the disputed work 哀弦篇 
to determine their authorship.

IMPORTANT: You must respond ONLY with a valid JSON object.
Do not include ANY text before or after the JSON."""

    def _get_basic(self) -> str:
        """Return linguistic knowledge for zero-shot prompting."""
        return """Analyze the writing styles of the input texts, disregarding 
        differences in topic and content, and reason based on linguistic features 
        such as character and punctuation frequency."""

    def _get_knowledge(self) -> str:
        """Return linguistic knowledge for zero-shot prompting."""
        return """Features supporting Lu Xun's style include: 之, 而, 唯, 矣, 是, 于是, 
足以, 必, 何, 徒, 然, 不, 乃, 于, 则, 进而, 全, 光, 夫.

Features supporting Zhou Zuoren's style include: 本, 及, 别, 原, 各, 为, 多, 但, 自然, 随.

Use these linguistic markers to analyze the passage and justify your decision."""

    def _get_examples(self, train_data: List[Dict], num_examples: int) -> str:
        """Generate few-shot examples from training data."""
        authors = ["lx", "zzr"]
        examples_per_author = {author: [] for author in authors}

        for d in train_data:
            if d["author"] in authors:
                examples_per_author[d["author"]].append(d)

        selected_examples = []
        for author in authors:
            if examples_per_author[author]:
                selected_examples.append(random.choice(examples_per_author[author]))

        remaining_slots = num_examples - len(selected_examples)
        if remaining_slots > 0:
            all_examples = [ex for exs in examples_per_author.values() for ex in exs]
            if all_examples:
                selected_examples.extend(random.sample(all_examples,
                                                       min(remaining_slots,
                                                           len(all_examples))))

        return "\n".join([
            f"Example {i + 1}:\nText: {ex['text']}\nAuthor: {ex['author'].upper()}\n"
            for i, ex in enumerate(selected_examples)
        ])

    def _get_output_format(self, stage: str) -> str:
        """Get the expected output format based on the stage."""
        if stage == "cot":
            return """{
    "analysis": "Your step-by-step reasoning about the stylometric features and decision",
    "author": "LX" or "ZZR"
}"""
        else:
            return """{
    "author": "LX" or "ZZR"
}"""

    def construct_prompt(self, text: str, stage: str,
                         train_data: Optional[List[Dict]] = None,
                         num_examples: int = 0) -> Dict[str, str]:
        """Construct the full prompt based on stage and parameters.

        Returns:
            Dict with 'system' and 'user' prompts separately.
        """
        # build the user prompt piece by piece
        user_prompt_parts = []

        # start with format instructions
        user_prompt_parts.append(
            f"Required JSON format:\n{self._get_output_format(stage)}")

        # ignore content (Huang et al. 2024)
        user_prompt_parts.append(self._get_basic())

        # add zero-shot knowledge if applicable
        if stage in ["zero-shot", "few-shot", "cot"]:
            user_prompt_parts.append("Linguistic Features to Consider:")
            user_prompt_parts.append(self._get_knowledge())

        # add examples if applicable
        if stage in ["few-shot", "cot"] and train_data and num_examples > 0:
            user_prompt_parts.append("Reference Examples:")
            user_prompt_parts.append(self._get_examples(train_data, num_examples))

        # add CoT-specific instructions if applicable
        if stage == "cot":
            user_prompt_parts.append(
                "Please think aloud and reason the likely author step by step based on "
                "character and punctuation frequency before reaching a conclusion."
            )

        # add the text to analyze at the end
        user_prompt_parts.append("Text to Analyze:")
        user_prompt_parts.append(text)

        return {
            "system": self.system_msg,
            "user": "\n\n".join(user_prompt_parts)
        }


class ModelManager:
    """Manage different LLM backends (OpenAI API and local models)."""

    def __init__(self, model_name: str, temperature: float = 0.6, seed: int = 42,
                 max_retries: int = 5):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.is_openai = model_name.startswith("gpt-")

        if self.is_openai:
            assert "OPENAI_API_KEY" in os.environ, "OpenAI API key not found"
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            self.model = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=2 if "70B" in model_name else 1,
                # use 2 GPUs for 70B
                enforce_eager=True,
                max_num_batched_tokens=4096,
                quantization="8bit" if "70B" in model_name else None,
                # add quantization for 70B
                device="cuda",
            )

    def _is_valid_response(self, response: Dict[str, Any]) -> bool:
        """Validate if the response has required fields and valid values."""
        if not isinstance(response, dict):
            return False

        # check required fields
        if "author" not in response:
            return False

        # validate author value
        if response["author"] not in ["LX", "ZZR"]:
            return False

        return True

    def _format_chat_messages(self, prompt_dict: Dict[str, str]) -> List[
        Dict[str, str]]:
        """Format prompts into chat messages."""
        return [
            {"role": "system", "content": prompt_dict["system"]},
            {"role": "user", "content": prompt_dict["user"]}
        ]

    def _generate_single(self, prompt_dict: Dict[str, str], run_seed: int) -> Optional[
        Dict[str, Any]]:
        """Single generation attempt with simplified response handling."""
        try:
            if self.is_openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=self._format_chat_messages(prompt_dict),
                    temperature=self.temperature,
                    seed=run_seed
                )
                raw_response = response.choices[0].message.content
            else:
                # create new SamplingParams for each run
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=1024 * 4,
                    seed=run_seed
                )

                # format as chat messages
                chat_messages = self._format_chat_messages(prompt_dict)

                # generate using chat API
                outputs = self.model.chat(
                    messages=[chat_messages],  # wrap in list for single request
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                raw_response = outputs[0].outputs[0].text

            # clean up and parse response
            raw_response = raw_response.strip()

            # find the complete JSON object
            start_idx = raw_response.find('{')
            if start_idx == -1:
                print("No JSON object found in response")
                return None

            raw_response = raw_response[start_idx:]
            end_idx = raw_response.find('}')
            if end_idx == -1:
                print("No closing brace found in response")
                return None

            # take the first complete JSON object
            raw_response = raw_response[:end_idx + 1]

            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError as e:
                print("=" * 90)
                print(f"JSON Parsing Failed. Error: {str(e)}")
                print("Raw response:")
                print(raw_response)
                print("=" * 90)
                return None

            if self._is_valid_response(result):
                return result
            else:
                print("=" * 90)
                print(f"Invalid response structure:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("=" * 90)
                return None

        except Exception as e:
            print("=" * 90)
            print(f"Generation error ({type(e).__name__}): {str(e)}")
            print("=" * 90)
            return None

    def generate_with_retries(self, prompt_dict: Dict[str, str],
                              required_results: int) -> List[Dict[str, Any]]:
        valid_results = []
        attempt = 0
        run_id = 1073
        errors = []

        while len(
                valid_results) < required_results and attempt < required_results * self.max_retries:
            run_seed = self.seed + run_id
            result = self._generate_single(prompt_dict, run_seed)

            if result is not None:
                valid_results.append(result)
            else:
                errors.append(f"Failed attempt {attempt + 1} with seed {run_seed}")

            attempt += 1
            run_id += 1

        success_rate = len(valid_results) / attempt if attempt > 0 else 0
        if errors:
            print(f"\nGeneration Statistics:")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Total attempts: {attempt}")
            print(f"Valid results: {len(valid_results)}")
            print(f"First few errors: {errors[:3]}")

        return valid_results


def compute_vote_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated results from multiple runs.
    Assumes all results are valid (no ERROR states).
    """
    possible_authors = ["LX", "ZZR"]

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
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name (Hugging Face or OpenAI)")
    parser.add_argument("--stage", choices=["basic", "zero-shot", "few-shot", "cot"],
                        required=True, help="Analysis stage to run")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples (for few-shot and cot stages)")
    parser.add_argument("--num_runs", type=int, default=30,
                        help="Number of runs for voting")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    # init
    random.seed(args.seed)
    train, val, test = load_corpus()
    model_mgr = ModelManager(args.model, args.temperature, args.seed)
    prompt_mgr = PromptManager(use_cot=(args.stage == "cot"))
    logger = ExperimentLogger()

    # first run validation set
    print("\nRunning validation set evaluation...")
    val_results = []
    for val_doc in val:
        print(f"\nValidating on document: {val_doc['title']}")
        prompt = prompt_mgr.construct_prompt(
            val_doc["text"],
            stage=args.stage,
            train_data=train if args.stage in ["few-shot", "cot"] else None,
            num_examples=args.num_examples if args.stage in ["few-shot", "cot"] else 0
        )
        run_results = model_mgr.generate_with_retries(prompt, args.num_runs)
        val_results.append({
            'doc': val_doc,
            'results': run_results,
            'aggregated_result': compute_vote_results(run_results)
        })

    # then run test set
    print("\nRunning test set prediction...")
    test_results = []
    for test_doc in test:
        print(f"\nAnalyzing document: {test_doc['title']}")
        prompt = prompt_mgr.construct_prompt(
            test_doc["text"],
            stage=args.stage,
            train_data=train if args.stage in ["few-shot", "cot"] else None,
            num_examples=args.num_examples if args.stage in ["few-shot", "cot"] else 0
        )
        run_results = model_mgr.generate_with_retries(prompt, args.num_runs)
        test_results.append({
            'doc': test_doc,
            'results': run_results,
            'aggregated_result': compute_vote_results(run_results)
        })

    # save the single configuration results
    config = vars(args).copy()
    # use the last prompt as example (they're all the same except for the text)
    logger.save_experiment(config, val, test, prompt, val_results, test_results)


if __name__ == "__main__":
    main()
