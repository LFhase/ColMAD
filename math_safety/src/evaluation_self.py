import json
import numpy as np
import pandas as pd
import re
import argparse
import os
import sys
import glob
from collections import Counter

from src.math_parser import extract_math_answer as extract_answer, compare_answers as math_equal

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "collegemath", "math500", "aime2024", "aime2025"], 
                        help="Dataset to evaluate (gsm8k, collegemath, math500, aime2024, aime2025)")
    parser.add_argument("--sc", action="store_true", default=True, help="Use majority voting for answer selection")
    return parser.parse_args()

def parse_answer(input_str):
    """Extract numerical answer from a string."""
    pattern = r"([0-9]*)"
    matches = re.findall(pattern, input_str)

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            return str(solution)
    
    return None

def most_frequent(answers):
    """Return the most frequently occurring answer in a list, using mathematical equivalence."""
    answers = [a for a in answers if a is not None]  # Remove None values

    if not answers:
        return None
    
    # Custom frequency counting with math_equal
    freq = {}
    for ans in answers:
        matched = False
        for known_ans in freq:
            if math_equal(ans, known_ans):
                freq[known_ans] += 1
                matched = True
                break
        if not matched:
            freq[ans] = 1
            
    # Find the most frequent answer
    return max(freq.items(), key=lambda x: x[1])[0]

def compute_accuracy(gt, pred_solutions, use_majority, dataset, math500_dataset=None, question_id=None):
    """Compare the predicted solution with the ground truth."""
    # For math500, we need to get the solution from the dataset
    if dataset == "math500":
        try:
            # Get the solution from the dataset using question_id as index
            gt_answer = extract_answer(math500_dataset[int(question_id)]["solution"], dataset)
        except (IndexError, ValueError):
            # Fallback to using the answer field if dataset loading fails
            gt_answer = gt
    else:
        try:
            # Try to convert gt to float directly
            gt_answer = float(gt)
        except (ValueError, TypeError):
            # If conversion fails, use extract_answer
            gt_answer = extract_answer(gt, dataset)

    if gt_answer is None or not pred_solutions:
        return None

    if use_majority:
        pred_answer = most_frequent(pred_solutions)  # Apply majority voting
    else:
        pred_answer = pred_solutions[0]  # Default: use first answer

    return 1.0 if math_equal(pred_answer, gt_answer) else 0.0 if pred_answer is not None else None

def evaluate_results(data, use_majority, dataset, math500_dataset=None, sample_size=None):
    """Evaluate results for a single model and single round."""
    model_name = list(data[0]["agent_response"].keys())[0]  # Get the model name
    results = []
    accuracies = []
    
    # Check if this is self-refinement by looking at the response format
    is_self_refinement = len(data[0]["agent_response"][model_name][0]) == 1
    
    for entry in data:
        question_id = entry["question_id"]
        gt = entry["answer"]
        responses = entry["agent_response"].get(model_name, [[]])
        
        if is_self_refinement:
            # For self-refinement, responses is a list of lists where each inner list contains one response
            # We want to evaluate only the last response up to the current step
            if responses:  # Check if there are any responses
                last_response = responses[-1]  # Get the last response
                if last_response:  # Check if the last response is not empty
                    solution = extract_answer(last_response[0], dataset)
                    pred_solutions = [solution] if solution is not None else []
                else:
                    pred_solutions = []
            else:
                pred_solutions = []
        else:
            # For self-consistency, responses[0] is a list of n responses
            # If sample_size is specified, only use the first n samples
            if sample_size is not None:
                responses = [responses[0][:sample_size]]
            pred_solutions = [extract_answer(ans, dataset) for ans in responses[0]]

        accuracy = compute_accuracy(gt, pred_solutions, use_majority, dataset, math500_dataset, question_id)
        if accuracy is not None:
            accuracies.append(accuracy)

        # Store results
        result_entry = {
            "question_id": question_id,
            model_name: "✅" if accuracy == 1 else "❌" if accuracy == 0 else "⚠️"
        }
        results.append(result_entry)

    return model_name, accuracies, results

def save_results(output_dir, model_name, accuracies, results, level=None, sample_size=None):
    """Save evaluation results to files."""
    acc_summary = {
        model_name: np.mean(accuracies) if accuracies else 0
    }
    
    # Add sample size to filename if specified
    sample_suffix = f"_sample_{sample_size}" if sample_size is not None else ""
    
    if level is not None:
        with open(f"{output_dir}/accuracy_summary_level{level}{sample_suffix}.json", "w") as f:
            json.dump(acc_summary, f, indent=4)
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/accuracy_results_level{level}{sample_suffix}.csv", index=False)
    else:
        with open(f"{output_dir}/accuracy_summary{sample_suffix}.json", "w") as f:
            json.dump(acc_summary, f, indent=4)
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/accuracy_results{sample_suffix}.csv", index=False)

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

def get_max_sample_size(data):
    """Get the maximum number of responses available in the data."""
    model_name = list(data[0]["agent_response"].keys())[0]
    max_responses = 0
    for entry in data:
        responses = entry["agent_response"].get(model_name, [[]])
        max_responses = max(max_responses, len(responses[0]))
    return max_responses

def main():
    args = args_parse()
    dataset = args.dataset
    base_dir = f"results/{dataset}"
    
    # Get all directories starting with "1_" in the dataset directory
    dirs = [d for d in glob.glob(f"{base_dir}/1_*") if os.path.isdir(d)]
    
    if not dirs:
        print(f"No directories starting with '1_' found in {base_dir}")
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
    
    for input_dir in dirs:
        print(f"\nProcessing {input_dir}...")
        conv_path = os.path.join(input_dir, 'conversation.json')
        
        # Skip if conversation.json doesn't exist
        if not os.path.exists(conv_path):
            print(f"  Skipping {input_dir} as conversation.json doesn't exist")
            continue
            
        # # skip if accuracy_summary.json exists
        # if os.path.exists(os.path.join(input_dir, 'accuracy_summary.json')):
        #     print(f"  Skipping {input_dir} as accuracy_summary.json already exists")
        #     continue
        
        # Load data and add levels for math500 if needed
        if dataset.lower() == "math500":
            data = add_math500_levels(conv_path)
        else:
            with open(conv_path, "r") as f:
                data = json.load(f)
        
        # Check if this is self-refinement by looking at directory name
        is_self_refinement = "sr_" in input_dir
        
        if is_self_refinement:
            # For self-refinement, evaluate different refinement steps
            print("Evaluating self-refinement responses...")
            # Get the maximum number of refinement steps
            max_steps = max(len(entry["agent_response"][list(entry["agent_response"].keys())[0]]) for entry in data)
            print(f"Maximum number of refinement steps: {max_steps}")

            sample_sizes = [2**i for i in range(0, int(np.log2(max_steps)) + 1)]
            
            if dataset.lower() == "math500":
                # Group the data by level for MATH500
                print("Starting evaluation for Math500...")
                levels = set(entry.get("level", None) for entry in data)
                if None in levels:
                    levels.remove(None)
                
                for level in levels:
                    print(f"Evaluating level {level}...")
                    # Filter the data by level
                    level_data = [entry for entry in data if entry.get("level") == level]
                    
                    # Evaluate specific refinement steps
                    for step in sample_sizes:
                        if step <= max_steps:
                            print(f"  Evaluating refinement step {step}...")
                            # Modify data to only use responses up to the specified step
                            step_data = []
                            for entry in level_data:
                                entry_copy = entry.copy()
                                model_name = list(entry["agent_response"].keys())[0]
                                entry_copy["agent_response"] = {
                                    model_name: entry["agent_response"][model_name][:step]
                                }
                                step_data.append(entry_copy)
                            
                            model_name, accuracies, results = evaluate_results(step_data, False, args.dataset, math500_dataset)
                            save_results(input_dir, model_name, accuracies, results, level=level, sample_size=step)
                
                print("Evaluating all levels...")
                # Evaluate final refinement step for all levels
                model_name, accuracies, results = evaluate_results(data, args.sc, args.dataset, math500_dataset)
                save_results(input_dir, model_name, accuracies, results)
            else:
                # Evaluate specific refinement steps (2nd and 4th)
                for step in sample_sizes:
                    if step <= max_steps:
                        print(f"  Evaluating refinement step {step}...")
                        # Modify data to only use responses up to the specified step
                        step_data = []
                        for entry in data:
                            entry_copy = entry.copy()
                            model_name = list(entry["agent_response"].keys())[0]
                            entry_copy["agent_response"] = {
                                model_name: entry["agent_response"][model_name][:step]
                            }
                            step_data.append(entry_copy)
                        
                        model_name, accuracies, results = evaluate_results(step_data, False, args.dataset, math500_dataset)
                        save_results(input_dir, model_name, accuracies, results, sample_size=step)
                
                # Evaluate final refinement step
                print("  Evaluating final refinement step...")
                model_name, accuracies, results = evaluate_results(data, args.sc, args.dataset, math500_dataset)
                save_results(input_dir, model_name, accuracies, results)
        else:
            # For self-consistency, evaluate with different sample sizes
            # Get maximum number of responses
            max_samples = get_max_sample_size(data)
            print(f"Maximum number of responses: {max_samples}")
            
            # Calculate sample sizes (powers of 2 up to max_samples, including 1)
            sample_sizes = [1] + [2**i for i in range(1, int(np.log2(max_samples)) + 1)]
            print(f"Evaluating with sample sizes: {sample_sizes}")
            
            # Evaluate and save results
            if dataset.lower() == "math500":
                # Group the data by level for MATH500
                print("Starting evaluation for Math500...")
                levels = set(entry.get("level", None) for entry in data)
                if None in levels:
                    levels.remove(None)
                
                for level in levels:
                    print(f"Evaluating level {level}...")
                    # Filter the data by level
                    level_data = [entry for entry in data if entry.get("level") == level]
                    
                    # Evaluate for each sample size
                    for sample_size in sample_sizes:
                        print(f"  Evaluating with {sample_size} samples...")
                        model_name, accuracies, results = evaluate_results(level_data, args.sc, args.dataset, math500_dataset, sample_size)
                        save_results(input_dir, model_name, accuracies, results, level=level, sample_size=sample_size)
                
                print("Evaluating all levels...")
                
                # Total evaluation for each sample size
                for sample_size in sample_sizes:
                    print(f"  Evaluating with {sample_size} samples...")
                    model_name, accuracies, results = evaluate_results(data, args.sc, args.dataset, math500_dataset, sample_size)
                    save_results(input_dir, model_name, accuracies, results, None, sample_size)
            else:
                # Evaluate and save results for GSM8K, CollegeMath, and AIME2024
                for sample_size in sample_sizes:
                    print(f"  Evaluating with {sample_size} samples...")
                    model_name, accuracies, results = evaluate_results(data, args.sc, args.dataset, math500_dataset, sample_size)
                    save_results(input_dir, model_name, accuracies, results, sample_size=sample_size)
        
        print(f"  ✅ Evaluation complete for {input_dir}")
    
    print(f"\n✅ All evaluations complete for {args.dataset}!")

if __name__ == "__main__":
    main() 