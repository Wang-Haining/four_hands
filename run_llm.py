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


class ExperimentLogger:
    """Handles logging of experimental results."""

    def __init__(self):
        self.output_dir = Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_id(self, config: Dict) -> str:
        """Generate unique experiment identifier."""
        return (f"{config['model'].split('/')[-1]}_"
                f"stage-{config['stage']}_"
                f"temp{config['temperature']}_"
                f"seed{config['seed']}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def save_experiment(self, config: Dict, prompt: str, results: List[Dict]):
        """Save single experiment results."""
        experiment_id = self.get_experiment_id(config)

        # organize experiment data
        experiment_data = {
            "config": config,
            "prompt": prompt,
            "individual_results": results,
            "aggregated_result": compute_vote_results(results)
        }

        # save to JSON file
        output_file = self.output_dir / f"{experiment_id}.json"
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

        print(f"\nExperiment results saved to: {output_file}")
        self._print_experiment_summary(experiment_data)


class PromptManager:
    """Manage different prompting strategies for authorship analysis."""

    def __init__(self, use_cot: bool = False):
        self.use_cot = use_cot
        self.system_msg = """You are an expert in Chinese literature, specializing in 
        stylometric analysis. You will analyze a passage from the disputed work 哀弦篇 to 
        determine if it was written by Lu Xun (鲁迅), Zhou Zuoren (周作人), 
        or was a collaboration between the brothers.

        IMPORTANT: You must ONLY respond with a valid JSON object. Do not include ANY 
        text before or after the JSON. The JSON must contain the required fields and 
        end with a single closing brace.
        """

    def _get_knowledge(self) -> str:
        """Return linguistic knowledge for zero-shot prompting."""
        return f"""Consider these 31 key linguistic features that distinguish between 
        the authors. Features supporting Lu Xun's style include: 之, 而, 唯, 矣, 是, 于是, 
        足以, 必, 何, 徒, 然, 不, 乃, 于, 则, 进而, 全, 光, 夫. Features supporting Zhou 
        Zuoren's style include: 本, 及, 别, 原, 各, 为, 多, 但, 自然, 随. Use this knowledge 
        to analyze the given disputed passage, identify its likely author, and justify 
        your decision by referencing relevant linguistic features."""

    def _get_examples(self, train_data: List[Dict], num_examples: int) -> str:
        """Generate few-shot examples from training data."""
        authors = ["lx", "zzr", "collab"]
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

        return "\n".join([f"Text: {ex['text']}\nAuthor: {ex['author'].upper()}\n"
                          for ex in selected_examples])

    def _get_output_format(self, stage: str) -> str:
        """Get the expected output format based on the stage."""
        if stage == "cot":
            return """{
    "analysis": "Provide detailed step-by-step reasoning about the stylometric features 
    you identified and how they led to your decision",
    "author": "LX" or "ZZR" or "COLLAB"
}"""
        else:
            return """{
    "author": "LX" or "ZZR" or "COLLAB"
}"""

    def construct_prompt(self, text: str, stage: str,
                         train_data: Optional[List[Dict]] = None,
                         num_examples: int = 0) -> str:
        """Construct the full prompt based on stage and parameters."""
        # start with system message
        prompt = f"{self.system_msg}\n\n"

        # add output format instructions
        prompt += f"Your response must match this JSON format:\n{self._get_output_format(stage)}\n\n"

        # add the text to analyze
        prompt += f"Analyze this text:\n{text}\n\n"

        # add zero-shot knowledge if applicable
        if stage in ["zero-shot", "few-shot", "cot"]:
            prompt += f"Use this linguistic knowledge:\n{self._get_knowledge()}\n\n"

        # add examples if applicable
        if stage in ["few-shot", "cot"] and train_data and num_examples > 0:
            prompt += f"Here are some examples:\n{self._get_examples(train_data, num_examples)}\n\n"

        # add CoT-specific instructions if applicable
        if stage == "cot":
            prompt += """Please think aloud and reason step by step and include this analysis in your response."""

        return prompt


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
                gpu_memory_utilization=0.8,
                tensor_parallel_size=1,
                enforce_eager=True,
                max_num_batched_tokens=1024*2,
                quantization=None,
                device="cuda",
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=1024*2,
                seed=seed
            )

    def _is_valid_response(self, response: Dict[str, Any]) -> bool:
        """Validate if the response has required fields and valid values."""
        if not isinstance(response, dict):
            return False

        # check required fields
        if "author" not in response:
            return False

        # validate author value
        if response["author"] not in ["LX", "ZZR", "COLLAB"]:
            return False

        return True

    def _generate_single(self, prompt: str, run_seed: int) -> Optional[Dict[str, Any]]:
        """Single generation attempt with simplified response handling."""
        try:
            if self.is_openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You must ONLY output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    seed=run_seed
                )
                raw_response = response.choices[0].message.content
            else:
                self.sampling_params.seed = run_seed
                outputs = self.model.generate([prompt], self.sampling_params)
                raw_response = outputs[0].outputs[0].text

            # clean up first
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

    def generate_with_retries(self, prompt: str, required_results: int) -> List[Dict[str, Any]]:
        """
        Generate responses until required number of valid results is obtained.

        Args:
            prompt: The input prompt
            required_results: Number of valid results needed

        Returns:
            List of valid results
        """
        valid_results = []
        attempt = 0
        run_id = 0

        while len(valid_results) < required_results and attempt < required_results * self.max_retries:
            run_seed = self.seed + run_id
            result = self._generate_single(prompt, run_seed)

            if result is not None:
                valid_results.append(result)

            attempt += 1
            run_id += 1

        if len(valid_results) < required_results:
            print(f"Warning: Only obtained {len(valid_results)} valid results "
                  f"out of {required_results} requested after {attempt} attempts")

        return valid_results


def compute_vote_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated results from multiple runs.
    Assumes all results are valid (no ERROR states).
    """
    possible_authors = ["LX", "ZZR", "COLLAB"]

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

    # run analysis for specified stage
    for test_doc in test:
        print(f"\nAnalyzing document: {test_doc['title']}")

        prompt = prompt_mgr.construct_prompt(
            test_doc["text"],
            stage=args.stage,
            train_data=train if args.stage in ["few-shot", "cot"] else None,
            num_examples=args.num_examples if args.stage in ["few-shot", "cot"] else 0
        )

        # run multiple times for voting
        run_results = model_mgr.generate_with_retries(prompt, args.num_runs)

        # save results
        config = vars(args).copy()
        config["test_doc"] = test_doc["title"]
        logger.save_experiment(config, prompt, run_results)


if __name__ == "__main__":
    main()
