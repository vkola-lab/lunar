# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict
import random

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask
from collections import Counter, defaultdict


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None
    
def extract_choice_description(choices_str: str, target_letter: str) -> str:
    """
    Extracts the description corresponding to the given letter from a formatted choices string.
    Example: If target_letter is 'A', return 'Normal Cognition'
    """
    pattern = rf"{target_letter.upper()}:\s*(.*?)(?=\n[A-Z]:|\Z)"
    match = re.search(pattern, choices_str, re.DOTALL)
    return match.group(1).strip() if match else ""

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


# def correctness_reward(completions, ground_truth, options, ID, return_answers=False, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has the answer."""
    
#     def extract_answer(text):
#         answer_match = re.search(r'<answer>\n(Answer:\s*)([a-zA-Z])\n</answer>', text, re.DOTALL)
#         if not answer_match:
#             return None
#         return answer_match.group(2).strip().lower()

#     # Extract answers
#     contents = [completion[0]["content"] for completion in completions]
#     answers = [extract_answer(content) for content in contents]
    
#     rewards = [
#         1.0 if ans == gt.lower() else 0.0 
#         for ans, gt in zip(answers, ground_truth)
#     ]
    
#     if return_answers:
#         return rewards, answers
        
#     return rewards

def correctness_reward(completions, ground_truth, return_answers=False, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the answer."""
    
    def extract_answer(text):
        boxed_match = re.search(r'<answer>.*\\boxed{(.*?)}.*</answer>', text, re.DOTALL)
        if not boxed_match:
            return None
        return boxed_match.group(1).strip().lower()

    # Extract answers
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(content) for content in contents]
    
    rewards = [
        1.0 if ans == gt.lower() else 0.0 
        for ans, gt in zip(answers, ground_truth)
    ]
    
    if return_answers:
        return rewards, answers
        
    return rewards


def majority_voting_reward(completions, ID, **kwargs) -> list[float]:
    """Reward function that checks if each completion has the majority-voted answer in its own ID group."""

    def extract_answer(text):
        answer_match = re.search(r'<answer>\n(Answer:\s*)([a-zA-Z])\n</answer>', text, re.DOTALL)
        if not answer_match:
            return None
        return answer_match.group(2).strip().lower()

    # Extract answers
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(content) for content in contents]

    # Group answers by ID
    id_to_answers = defaultdict(list)
    for idx, id_val in enumerate(ID):
        id_to_answers[id_val].append(answers[idx])
        
    print(id_to_answers)

    # Compute majority answer per ID
    id_to_majority = {}
    for id_val, group_answers in id_to_answers.items():
        filtered_group_answers = [ans for ans in group_answers if ans is not None]
        counts = Counter(filtered_group_answers)
        majority_answer, _ = counts.most_common(1)[0]
        id_to_majority[id_val] = majority_answer
        
    print(id_to_majority)

    # Assign rewards based on each completion's ID group majority
    rewards = [
        1.0 if ans == id_to_majority[id_val].lower() else 0.0
        for ans, id_val in zip(answers, ID)
    ]
    
    return rewards

def hybrid_reward(completions, ground_truth, options, ID, **kwargs) -> list[float]:
    """Hybrid reward: if all answers for an ID are wrong, pick one randomly to have reward 1; else use correctness."""
    
    # Compute correctness rewards
    correctness_rewards, extracted_answers = correctness_reward(completions, ground_truth, options, ID, return_answers=True)

    # Group correctness rewards by ID
    id_to_correctness = defaultdict(list)
    id_to_answers = defaultdict(list)
    id_to_indices = defaultdict(list)
    for idx, (reward, answer, id_val) in enumerate(zip(correctness_rewards, extracted_answers, ID)):
        id_to_correctness[id_val].append(reward)
        id_to_answers[id_val].append(answer)
        id_to_indices[id_val].append(idx)

    # Decide per ID
    rewards = correctness_rewards.copy()  # start with correctness rewards
    for id_val, group_rewards in id_to_correctness.items():
        if all(r == 0.0 for r in group_rewards) and not any(ans is None for ans in id_to_answers[id_val]): # all rewards are zero and no None in the extracted answers - this makes sure that random index gets flipped only when all the answers are wrong
            print(f"Flipped random index for ID: {id_val}")
            # print(f"Before: {correctness_rewards}")
            # pick one random index for this ID
            indices = id_to_indices[id_val]
            chosen_idx = random.choice(indices)
            rewards[chosen_idx] = 1.0  # set reward 1 for this one
            # print(f"After: {rewards}")

    # Debug prints
    print(id_to_answers)
    print(id_to_correctness)
    id_to_correctness_after = defaultdict(list)
    for idx, (reward, id_val) in enumerate(zip(rewards, ID)):
        id_to_correctness_after[id_val].append(reward)
        
    print(id_to_correctness_after)
    return rewards

# https://github.com/knoveleng/open-rs/blob/main/src/open_r1/rewards.py
def english_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is in English."""
    import unicodedata
    from langdetect import detect, LangDetectException

    def is_non_english(text):
        """
        Checks if the given text contains languages other than English.
        Ignores LaTeX notation.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            bool: False if the text is in English (with LaTeX allowed),
                True if it contains non-English languages
        """
        # Skip if empty
        if not text or text.strip() == "":
            return False
        
        # First, remove LaTeX notation to avoid false positives
        # This pattern matches typical LaTeX structures like $...$ or \begin{...}...\end{...}
        # latex_pattern = r'\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\}'
        # text_without_latex = re.sub(latex_pattern, '', text, flags=re.DOTALL)
        
        # # Also remove common LaTeX commands
        # latex_commands = r'\\[a-zA-Z]+((\{[^{}]*\})?|(\[[^\[\]]*\])?)+'
        # text_without_latex = re.sub(latex_commands, '', text_without_latex)
        
        # Check if we have non-ASCII characters that are not typical in English text
        # First, normalize unicode characters
        normalized_text = unicodedata.normalize('NFKD', text)
        
        # Common non-English character sets (excluding common punctuation and symbols)
        non_english_patterns = [
            # Cyrillic characters
            r'[\u0400-\u04FF]',
            # Chinese/Japanese/Korean characters
            r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]',
            # Arabic characters
            r'[\u0600-\u06FF]',
            # Hebrew characters
            r'[\u0590-\u05FF]',
            # Thai characters
            r'[\u0E00-\u0E7F]',
            # Greek characters
            # r'[\u0370-\u03FF]',
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, normalized_text):
                return True
        
        # If no obvious non-English characters found, try language detection
        # Clean text further - remove URLs, numbers, punctuation
        cleaned_text = re.sub(r'http\S+|www\S+|\d+|[^\w\s]', ' ', text)
        cleaned_text = ' '.join(cleaned_text.split())
        
        # Only perform language detection if we have enough text
        if len(cleaned_text.split()) >= 5:
            try:
                detected_lang = detect(cleaned_text)
                return detected_lang != 'en'
            except LangDetectException:
                # If detection fails, rely on character-based detection above
                pass
        
        # Default to assuming it's English
        return False

    contents = [completion[0]["content"] for completion in completions]
    return [0 if has_non_english(content) else 1 for content in contents]

# def hybrid_reward(completions, ground_truth, options, ID, **kwargs) -> list[float]:
#     """Hybrid reward: if all answers for an ID are wrong, use majority voting; else use correctness."""
    
#     # Compute both rewards
#     majority_rewards = majority_voting_reward(completions, ID)
#     correctness_rewards = correctness_reward(completions, ground_truth, options, ID)

#     # Group correctness rewards by ID
#     id_to_correctness = defaultdict(list)
#     for reward, id_val in zip(correctness_rewards, ID):
#         id_to_correctness[id_val].append(reward)

#     # Decide per ID
#     id_to_decision = {}
#     for id_val, group_rewards in id_to_correctness.items():
#         if all(r == 0.0 for r in group_rewards):
#             id_to_decision[id_val] = "majority"
#         else:
#             id_to_decision[id_val] = "gt"

#     # Final rewards: choose per ID
#     rewards = [
#         majority_rewards[idx] if id_to_decision[id_val] == "majority" else correctness_rewards[idx]
#         for idx, id_val in enumerate(ID)
#     ]
#     print(id_to_correctness)
#     print(id_to_decision)
#     print(rewards)
#     # raise ValueError

#     return rewards



# def correctness_reward(completions, ground_truth, options, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has the answer."""
    
#     contents = [completion[0]["content"] for completion in completions]
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise

#     rewards = []
#     for i, (c, gt) in enumerate(zip(contents, ground_truth)):
#         # gt_desc = extract_choice_description(options[i], gt)
        
#         answer_match = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        
#         if not answer_match:
#             rewards.append(0.0)
#             continue
            
#         # letter_match = re.search(r'Answer:\s*([A-Za-z])\s+', answer_match.group(1).strip())
#         word_match = re.search(r'Answer:\s*([A-Za-z])', answer_match.group(1).strip())
        
#         if word_match and word_match.group(1).strip().lower() == gt.lower():
#             rewards.append(1.0)
#         # elif word_match and word_match.group(1).lower() == gt_desc.lower():
#         #     rewards.append(0.5)
#         else:
#             rewards.append(0.0)

        
#     return rewards



# def correctness_reward(completions, ground_truth, options, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has the answer."""
    
#     contents = [completion[0]["content"] for completion in completions]
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise

#     rewards = []
#     for i, (c, gt) in enumerate(zip(contents, ground_truth)):
            
#         # letter_match = re.search(r'Answer:\s*([A-Za-z])\s+', answer_match.group(1).strip())
#         word_match = re.search(r'Answer:\s*([A-Za-z])', c)
#         # print(c, gt, word_match)
#         # raise ValueError
        
#         if word_match and word_match.group(1).strip().lower() == gt.lower():
#             rewards.append(1.0)
#         else:
#             rewards.append(0.0)
            
        
#     return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards





def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, **kwargs)
    BINARY_THRESHOLD = 0.99
    return [1.0 if reward > BINARY_THRESHOLD else 0.0 for reward in rewards]


def code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "correctness": correctness_reward,
        "format": format_reward,
        "tag_count": tag_count_reward,
        "majority_voting": majority_voting_reward,
        "hybrid": hybrid_reward,
        "english": english_reward,
        "accuracy": accuracy_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(code_reward, num_parallel=script_args.parallel_code_exec_per_proc), code_reward
        ),
        "binary_code": update_wrapper(
            partial(binary_code_reward, num_parallel=script_args.parallel_code_exec_per_proc), binary_code_reward
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language)
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
