import json
import sys
import os
import numpy as np
import pandas as pd
import re
import argparse
import random
import glob

from src.math_parser import extract_math_answer as extract_answer, compare_answers as math_equal
random.seed(42)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["math500", "collegemath", "gsm8k", "aime2024", "aime2025"],help="Dataset name for evaluation")
    parser.add_argument("--folder", type=str, default="results")
    return parser.parse_args()


def most_frequent(lst):
    """Find the most frequent element in a list using mathematical equivalence."""
    if not lst:  # Handle empty list case
        return None
        
    # Count frequencies using math_equal for comparison
    freq = {}
    for item in lst:
        if item is not None:  # Skip None values
            matched = False
            for known_item in freq:
                if math_equal(item, known_item):
                    freq[known_item] += 1
                    matched = True
                    break
            if not matched:
                freq[item] = 1
            
    if not freq:  # If all items were None
        return None
        
    # Find the most frequent item
    most_freq_item = max(freq.items(), key=lambda x: x[1])[0]
    
    return most_freq_item


def compute_accuracy(gt, pred_solutions, dataset, math500_dataset=None, question_id=None):
    """Compute accuracy for predictions based on the dataset."""
    # For math500, we need to get the solution from the dataset
    if dataset == "math500":
        try:
            # Get the solution from the dataset using question_id as index
            gt_answer = extract_answer(math500_dataset[int(question_id)]["solution"], dataset)
        except (IndexError, ValueError):
            # Fallback to using the answer field if dataset loading fails
            raise ValueError("Failed to load Math500 dataset")
    elif dataset == "aime2024":
        gt_answer = gt
    else:
        gt_answer = extract_answer(gt, dataset)
    
    if gt_answer is None:
        return [None] * len(pred_solutions)
    
    pred_answers = [extract_answer(ans, dataset) for ans in pred_solutions]
    return [1.0 if math_equal(ans, gt_answer) else 0.0 if ans is not None else None for ans in pred_answers]


def evaluate_results(data, total_rounds, dataset, math500_dataset=None, sample_size=None):
    """Evaluate results for different models and rounds."""
    model_names = list(data[0]["agent_response"].keys())
    num_models = len(model_names)
    
    # If sample_size is provided, calculate which round to evaluate up to
    if sample_size is not None:
        round_to_evaluate = min(sample_size // num_models, total_rounds)
    else:
        round_to_evaluate = total_rounds
    
    round_accuracies = {round_id: {model: [] for model in model_names} for round_id in range(round_to_evaluate)}
    diff_same_ratios = []
    diff_correct_ratios = []
    same_correct_ratios = []
    
    results = []

    # total answer
    best_total_acc = 0
    random_total_acc = 0
    
    for entry in data:
        question_id = entry["question_id"]
        gt = entry["answer"]
        agent_response = entry["agent_response"]
        
        # Get the actual number of rounds for this entry
        actual_rounds = min(len(next(iter(agent_response.values()))), round_to_evaluate)
        
        # Get answers up to the actual number of rounds
        model_answers = {model: [agent_response.get(model, [None] * actual_rounds)[r] for r in range(actual_rounds)] for model in model_names}
        
        # Accuracy calculation
        acc_matrix = {model: compute_accuracy(gt, model_answers[model], dataset, math500_dataset, question_id) for model in model_names}
        for r in range(actual_rounds):
            for model in model_names:
                if acc_matrix[model][r] is not None:
                    round_accuracies[r][model].append(acc_matrix[model][r])

        # Use the last available round for final evaluation
        final_r = actual_rounds - 1
        final_answer = [extract_answer(model_answers[model][final_r], dataset) for model in model_names]

        unique_answers = set(filter(None, final_answer))
        if unique_answers:
            diff_same_ratio = (len(unique_answers) - 1) / (len(model_names) - 1) if len(model_names) > 1 else 0
            diff_same_ratios.append(diff_same_ratio)

            corrects = [acc_matrix[model][final_r] for model in model_names if acc_matrix[model][final_r] is not None]
            if diff_same_ratio > 0:
                diff_correct_ratios.append(any(corrects))
            else:
                same_correct_ratios.append(all(corrects))
        
        # final answer - majority voting
        major_ans = most_frequent(final_answer)
        if major_ans is not None:
            best_acc = compute_accuracy(gt, [str(major_ans)], dataset, math500_dataset, question_id)[0]
            best_acc = 0.0 if best_acc is None else best_acc
        else:
            best_acc = 0.0

        # Random model selection for comparison
        sampled_model = random.choice(model_names)
        random_answer = agent_response[sampled_model][final_r]
        random_acc = compute_accuracy(gt, [random_answer], dataset, math500_dataset, question_id)[0]
        random_acc = 0.0 if random_acc is None else random_acc

        best_total_acc += best_acc
        random_total_acc += random_acc

        # Save results for this question
        result_entry = {"question_id": question_id}
        for r in range(actual_rounds):
            for model in model_names:
                if acc_matrix[model][r] is None:
                    result_entry[f"{model}_round{r+1}"] = "⚠️"
                else:
                    result_entry[f"{model}_round{r+1}"] = "✅" if acc_matrix[model][r] == 1 else "❌"
        results.append(result_entry)
    
    return round_accuracies, diff_same_ratios, diff_correct_ratios, same_correct_ratios, results, best_total_acc, random_total_acc


def save_results(output_dir, round_accuracies, diff_same_ratios, diff_correct_ratios, same_correct_ratios, 
                results, best_total_acc, random_total_acc, total_count, level=None, sample_size=None):
    """Save evaluation results to files."""
    # Accuracy summary
    acc_summary = {round_id: {model: np.mean(round_accuracies[round_id][model]) if round_accuracies[round_id][model] else 0 
                             for model in round_accuracies[round_id]} 
                  for round_id in round_accuracies}

    acc_summary["total_best"] = best_total_acc / total_count if total_count > 0 else 0
    acc_summary["total_random"] = random_total_acc / total_count if total_count > 0 else 0

    # Statistics summary
    stat_summary = {
        "Different answers %": np.mean(diff_same_ratios) * 100 if diff_same_ratios else 0,
        "Same answers %": (1 - np.mean(diff_same_ratios)) * 100 if diff_same_ratios else 0,
        "Different but at least one correct %": np.mean(diff_correct_ratios) * 100 if diff_correct_ratios else 0,
        "Different and both wrong %": (1 - np.mean(diff_correct_ratios)) * 100 if diff_correct_ratios else 0,
        "Same and correct %": np.mean(same_correct_ratios) * 100 if same_correct_ratios else 0,
        "Same and wrong %": (1 - np.mean(same_correct_ratios)) * 100 if same_correct_ratios else 0
    }

    # Create filename suffixes
    level_suffix = f"_level{level}" if level is not None else ""
    sample_suffix = f"_sample_{sample_size}" if sample_size is not None else ""
    
    with open(f"{output_dir}/accuracy_summary{level_suffix}{sample_suffix}.json", "w") as f:
        json.dump(acc_summary, f, indent=4)
    
    with open(f"{output_dir}/statistics_summary{level_suffix}{sample_suffix}.json", "w") as f:
        json.dump(stat_summary, f, indent=4)
    
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/accuracy_results{level_suffix}{sample_suffix}.csv", index=False)


def add_math500_levels(data_path):
    """Add difficulty levels to Math500 data."""
    # Load the conversation data
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Check if levels are already added
    if all("level" in entry for entry in data):
        return data
        
    try:
        # Try to import datasets - if it fails, fall back to naive level extraction
        from datasets import load_dataset
        
        print("  Loading Math500 dataset from HuggingFace...")
        math500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        math500_questions = {str(i): {"level": item["level"]} for i, item in enumerate(math500_dataset)}
        
        # Map each question_id to the level from Math500 dataset
        for entry in data:
            question_id = entry["question_id"]
            if str(question_id) in math500_questions:
                entry["level"] = math500_questions[str(question_id)]["level"]
            else:
                # Fallback: try to extract level from question_id
                parts = str(question_id).split("_")
                if len(parts) > 1 and parts[-1].isdigit():
                    entry["level"] = int(parts[-1])
                else:
                    entry["level"] = 0
    except ImportError:
        print("  Warning: 'datasets' package not found. Using naive level extraction.")
        # Fallback to naive level extraction
        for entry in data:
            question_id = entry.get("question_id", "")
            parts = str(question_id).split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                entry["level"] = int(parts[-1])
            else:
                entry["level"] = 0
    
    # Save the updated data
    with open(data_path, "w") as f:
        json.dump(data, f, indent=4)
    
    return data


def get_total_rounds_from_dir(dir_name):
    """Extract the total rounds from directory name."""
    split_dir = dir_name.split("_")
    try:
        round_idx = split_dir.index('r') + 1
        if round_idx < len(split_dir) and split_dir[round_idx].isdigit():
            return int(split_dir[round_idx])
    except (ValueError, IndexError):
        pass
    
    # Default to 4 rounds if not found
    return 4


def main():
    args = args_parse()
    dataset = args.dataset.lower()
    base_dir = f"{args.folder}/{dataset}"
    
    # Get all directories NOT starting with "1_" but with a number in the dataset directory
    all_dirs = [d for d in glob.glob(f"{base_dir}/[2-9]*") if os.path.isdir(d)]
    # Also include directories starting with 10 or higher
    all_dirs.extend([d for d in glob.glob(f"{base_dir}/[1-9][0-9]*") if os.path.isdir(d)])
    
    if not all_dirs:
        print(f"No multi-agent directories found in {base_dir}")
        return
    
    # Load Math500 dataset once if needed
    math500_dataset = None
    if dataset == "math500":
        try:
            from datasets import load_dataset
            print("Loading Math500 dataset...")
            math500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
            print("Math500 dataset loaded successfully")
        except ImportError:
            print("Warning: 'datasets' package not found. Will use answer field instead of solution.")
    
    for input_dir in all_dirs:
        print(f"\nProcessing {input_dir}...")
        
        # Skip if results already exist
        if os.path.exists(os.path.join(input_dir, 'accuracy_summary.json')):
            print(f"  Skipping {input_dir} as evaluation results already exist")
            continue
        
        # Try different conversation file names
        conv_file = None
        for file_name in ['conversation.json', 'conversation_lv.json']:
            file_path = os.path.join(input_dir, file_name)
            if os.path.exists(file_path):
                conv_file = file_path
                break
        
        if not conv_file:
            print(f"  Skipping {input_dir} as no conversation file exists")
            continue
        
        # Load data and add levels for math500 if needed
        if dataset == "math500":
            data = add_math500_levels(conv_file)
        else:
            with open(conv_file, "r") as f:
                data = json.load(f)
        
        # Get total rounds from directory name
        total_rounds = get_total_rounds_from_dir(os.path.basename(input_dir))
        print(f"  Detected {total_rounds} rounds")
        
        # Get number of models
        num_models = len(list(data[0]["agent_response"].keys()))
        print(f"  Detected {num_models} models")
        
        # Calculate sample sizes for each round
        sample_sizes = [r * num_models for r in range(1, total_rounds + 1)]
        print(f"  Evaluating sample sizes: {sample_sizes}")
        
        # Check if we need level-based evaluation
        if dataset == 'math500' and any('level' in entry for entry in data):
            # Group the data by level
            levels = sorted(set(entry.get("level", 0) for entry in data if "level" in entry))
            
            for level in levels:
                if level == 0:  # Skip entries with no level
                    continue
                    
                # Filter the data by level
                level_data = [entry for entry in data if entry.get("level") == level]
                
                if not level_data:
                    continue
                
                # Evaluate for each sample size
                for sample_size in sample_sizes:
                    print(f"  Evaluating level {level} with sample size {sample_size}...")
                    results = evaluate_results(level_data, total_rounds, dataset, math500_dataset, sample_size)
                    round_accuracies, diff_same_ratios, diff_correct_ratios, same_correct_ratios, result_entries, best_total_acc, random_total_acc = results
                    
                    # Save the results for this level and sample size
                    save_results(
                        input_dir, 
                        round_accuracies, 
                        diff_same_ratios, 
                        diff_correct_ratios, 
                        same_correct_ratios, 
                        result_entries, 
                        best_total_acc, 
                        random_total_acc,
                        len(level_data),
                        level=level,
                        sample_size=sample_size
                    )
                
                print(f"  ✅ Evaluation complete for level {level}")
        
        # Always do full evaluation on all data for each sample size
        for sample_size in sample_sizes:
            print(f"  Evaluating with sample size {sample_size}...")
            results = evaluate_results(data, total_rounds, dataset, math500_dataset, sample_size)
            round_accuracies, diff_same_ratios, diff_correct_ratios, same_correct_ratios, result_entries, best_total_acc, random_total_acc = results
            
            save_results(
                input_dir, 
                round_accuracies, 
                diff_same_ratios, 
                diff_correct_ratios, 
                same_correct_ratios, 
                result_entries, 
                best_total_acc, 
                random_total_acc,
                len(data),
                sample_size=sample_size
            )
        
        print(f"  ✅ Evaluation complete for {input_dir}")
    
    print(f"\n✅ All evaluations complete for {args.dataset}!")


if __name__ == "__main__":
    main() 
