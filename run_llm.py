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

import openai
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel, Field
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from vllm import LLM, SamplingParams

from utils import load_corpus


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Evaluate predictions against ground truth."""
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


class AuthorshipResult(BaseModel):
    """Schema for the authorship analysis result."""
    author: str = Literal['LX', 'ZZR']
    analysis: Optional[str] = None


class PromptManager:
    """Enhanced prompt manager that maintains all original features with better JSON handling."""

    def __init__(self, use_cot: bool = False):
        self.use_cot = use_cot
        self.result_schema = AuthorshipResult.model_json_schema()

        # base system prompt with JSON emphasis
        self.system_msg = """You are an expert in Chinese literature, specializing in 
stylometric analysis. Your task is to analyze passages from the disputed work 哀弦篇 
to determine their authorship."""

    def _get_basic(self) -> str:
        """Return basic analysis instructions."""
        return """Analyze the writing styles of the input texts, disregarding 
differences in topic and content, and reason based on linguistic features 
such as character frequency."""

    def _get_knowledge(self) -> str:
        """Return linguistic knowledge for zero-shot prompting."""
        return """Features supporting Lu Xun's style include: 之, 而, 唯, 矣, 是, 于是, 
足以, 必, 何, 徒, 然, 不, 乃, 于, 则, 进而, 全, 光, 夫.

Features supporting Zhou Zuoren's style include: 本, 及, 别, 原, 各, 为, 多, 但, 自然, 随."""

    def _get_examples(self, train_data: List[Dict], num_examples: int) -> str:
        """Generate few-shot examples with proper JSON formatting."""
        # get balanced examples for both authors
        lx_examples = [d for d in train_data if d['author'].upper() == 'LX']
        zzr_examples = [d for d in train_data if d['author'].upper() == 'ZZR']

        # ensure we get equal numbers of each author
        examples_per_author = num_examples // 2
        selected_lx = random.sample(lx_examples, examples_per_author)
        selected_zzr = random.sample(zzr_examples, examples_per_author)

        # if num_examples is odd, randomly add one more example
        if num_examples % 2 == 1:
            extra_example = random.choice(random.choice([lx_examples, zzr_examples]))
            selected_examples = selected_lx + selected_zzr + [extra_example]
        else:
            selected_examples = selected_lx + selected_zzr

        # shuffle the examples
        random.shuffle(selected_examples)

        # format examples
        formatted_examples = []
        for example in selected_examples:
            json_response = create_example_json(
                text="",  # we don't include the text in the response
                author=example['author'].upper()
            )
            formatted_examples.append(
                f"Text:\n{example['text']}\n\nResponse:\n{json_response}"
            )

        return "\n\n".join(formatted_examples)

    def _get_stage_instructions(self, stage: str) -> str:
        """Get stage-specific instructions."""
        if stage == "cot":
            return """Think through your analysis step by step. Consider the frequency 
of characteristic markers, sentence patterns, and stylistic choices. Include your 
reasoning in the 'analysis' field of your JSON response."""
        else:  # for basic, zero-shot, and few-shot scenarios
            return """Focus on low-level linguistic patterns and word choices to 
determine the likely author."""

    def construct_prompt(self, text: str, stage: str,
                         train_data: Optional[List[Dict]] = None,
                         num_examples: int = 0) -> Dict[str, str]:
        """Construct the full prompt maintaining all original features."""
        # start with the schema reminder
        prompt_parts = [
            f"You MUST answer using the following json schema: {AuthorshipResult.schema_json()}",
            self._get_basic()
        ]

        # add stage-specific content
        if stage in ["zero-shot", "few-shot", "cot"]:
            prompt_parts.append("Linguistic Features to Consider:")
            prompt_parts.append(self._get_knowledge())

        # add examples for few-shot and cot
        if stage in ["few-shot", "cot"] and train_data and num_examples > 0:
            prompt_parts.append("Reference Examples:")
            prompt_parts.append(self._get_examples(train_data, num_examples))

        # add stage-specific instructions
        prompt_parts.append(self._get_stage_instructions(stage))

        # add the text and final JSON reminder
        prompt_parts.extend([
            "Text to Analyze:",
            text,
        ])

        # build final prompt
        system_prompt = f"{self.system_msg}"

        return {
            "system": system_prompt,
            "user": "\n\n".join(prompt_parts)
        }


class ModelManager:
    """Model manager with robust JSON handling using lm-format-enforcer."""

    def __init__(self, model_name: str, temperature: float = 0.6, seed: int = 42):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.is_openai = model_name.startswith("gpt-")

        if not self.is_openai:
            # initialize local model with full parameters
            self.model = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=2 if "70B" in model_name else 1,
                enforce_eager=False,
                max_num_batched_tokens=8192,
                quantization="gptq" if "70B" in model_name else None,
                device="cuda",
            )

            # initialize JSON schema parser
            self.parser = JsonSchemaParser(AuthorshipResult.model_json_schema())

    def _format_messages(self, system_msg: str, user_msg: str) -> List[Dict[str, str]]:
        """Format chat messages."""
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

    def generate(self, prompt: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Generate a single valid response using lm-format-enforcer for JSON handling."""
        messages = self._format_messages(prompt["system"], prompt["user"])

        try:
            if self.is_openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                result = response.choices[0].message.content
                # validate with pydantic
                return AuthorshipResult.model_validate_json(result).model_dump()
            else:
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=8192
                )
                # use lm-format-enforcer for local models
                result = self.model.chat_with_parser(
                    messages=messages,
                    sampling_params=sampling_params,
                    parser=self.parser
                )
                # return first valid result
                return AuthorshipResult.model_validate_json(
                    result[0].outputs[0].text).model_dump()

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None

    def generate_with_retries(self, prompt: Dict[str, str], num_runs: int,
                              max_retries: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple responses with retry logic."""
        valid_results = []
        attempts = 0
        run_id = self.seed + 1000  # base for run seeds

        while len(valid_results) < num_runs and attempts < num_runs * max_retries:
            if self.is_openai:
                openai.api_key = os.environ["OPENAI_API_KEY"]

            result = self.generate(prompt)
            if result is not None:
                valid_results.append(result)

            attempts += 1
            run_id += 1

        success_rate = len(valid_results) / attempts if attempts > 0 else 0
        print(f"\nGeneration Statistics:")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Total attempts: {attempts}")
        print(f"Valid results: {len(valid_results)}")

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
