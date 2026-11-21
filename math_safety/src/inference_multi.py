import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from vllm import LLM, SamplingParams
from math_parser import extract_math_answer as extract_answer, compare_answers as math_equal
import json
import requests
import numpy as np
import re
import random
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
import aiohttp
import asyncio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import vLLM

# Import Hugging Face transformers for vector models

random.seed(42)
np.random.seed(42)
# Set more random seeds for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Maximum number of agents supported
MAX_AGENTS = 8

# vLLM API endpoints for different models
VLLM_API_URLS = {
    f"model_{i}": f"http://localhost:{8000 + i - 1}/v1/completions"
    for i in range(1, MAX_AGENTS + 1)
}


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, required=True, choices=range(2, MAX_AGENTS + 1),
                        help=f"Number of agents to use (2-{MAX_AGENTS})")
    for i in range(1, MAX_AGENTS + 1):
        parser.add_argument(f"--model_{i}", type=str, required=(i <= 2),
                            help=f"Model path for agent {i}")
        parser.add_argument(f"--gpu_{i}", type=int, default=i-1,
                            help=f"GPU index for agent {i}")
        parser.add_argument(f"--node_{i}", type=str,
                            help=f"Node name for agent {i}")
    parser.add_argument("--use_server", action="store_true", default=False,
                        help="Whether to use vLLM server APIs instead of direct integration")
    parser.add_argument("--huggingface", action="store_true", default=False,
                        help="Whether to use HuggingFace models for inference")
    parser.add_argument("--custom_cache", action="store_true", default=False,
                        help="Whether to use custom cache configuration with specified cache directories")
    parser.add_argument("--round", default=4, type=int)
    parser.add_argument("--num_query", default=100, type=int)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument("--early_stopping", default=True)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["gsm8k", "math500", "collegemath", "aime2024", "aime2025"],
        help="Dataset to use for inference"
    )
    # CollegeMath specific argument
    parser.add_argument(
        "--data_topic",
        default="algebra",
        type=str,
        choices=["algebra", "calculus", "precalculus", "differential_equation",
                 "linear_algebra", "probability", "vector_calculus"],
        help="Topic for CollegeMath dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the result file"
    )
    parser.add_argument(
        "--add_self_response",
        default=False,
        action='store_true',
        help="If True, models will receive their own previous responses in addition to other agents' responses"
    )

    for i in range(1, MAX_AGENTS + 1):
        parser.add_argument(f"--server_url_{i}", type=str, default=None,
                            help=f"Server URL for agent {i} (e.g. http://localhost:8000)")
    return parser.parse_args()


def prompt_formatting(instruction, cot, prompt_dict, model_path):
    model_list = prompt_dict.keys()
    model_name = model_path.split("/")[-1].split("-")[0].lower()

    for model_key in model_list:
        if model_key in model_name:
            model = model_key
            break
    if model == "alpaca" or model == "orca":
        formatted_prompt = prompt_dict[model]["prompt_no_input"].format(
            instruction=instruction)
    else:
        formatted_prompt = prompt_dict[model]["prompt"].format(
            instruction=instruction)

    if cot:
        formatted_prompt += " Let's think step by step."

    return formatted_prompt


def parse_answer(input_str, dataset_type):
    return extract_answer(input_str, dataset_type)


def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

# Load models directly (non-server mode)


def load_models(args):
    """
    Load one model per GPU and share it among agents assigned to the same GPU.
    """
    models = {}
    loaded_gpus = {}  # Maps GPU index to the loaded model

    # First, identify which models to load on which GPUs
    for i in range(1, args.num_agents + 1):
        model_key = f"model_{i}"
        model_path = getattr(args, model_key)
        gpu_idx = getattr(args, f"gpu_{i}")
        node = getattr(args, f"node_{i}")

        # Get current node from SLURM environment
        current_node = os.environ.get(
            "SLURM_NODELIST", "localhost").split(",")[0]
        if node and node != current_node:
            print(f"Skipping {model_path} as it should run on node {node}")
            continue

        # Only load the model if it hasn't been loaded on this GPU yet
        if gpu_idx not in loaded_gpus:
            # Set device
            device = f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"

            print(f"Loading {model_path} on GPU {gpu_idx}...")

            # Configure cache directories for custom cache if enabled
            if args.custom_cache:
                # Set cache directories for custom cache
                cache_dir = "CUSTOM_CACHE_DIR"  # Replace with your custom cache directory
                os.environ["TRANSFORMERS_CACHE"] = cache_dir
                os.environ["HF_DATASETS_CACHE"] = cache_dir
                print(f"Using custom cache directory: {cache_dir}")

                try:
                    # Use vLLM with cache directory
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    print(
                        f"Setting CUDA_VISIBLE_DEVICES={gpu_idx} for model {model_path}")
                    loaded_gpus[gpu_idx] = {
                        "model": LLM(
                            model=model_path,
                            tensor_parallel_size=1,
                            trust_remote_code=True,
                            dtype="bfloat16",
                            download_dir=cache_dir,
                            max_model_len=8192  # Set max sequence length
                        ),
                        "type": "vllm"
                    }
                    print(
                        f"Successfully loaded vLLM model {model_path} on GPU {gpu_idx} with custom cache")
                except Exception as e:
                    print(f"Error loading vLLM model {model_path}: {e}")
                    raise e
            elif args.huggingface:
                try:
                    # Load with Hugging Face
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map=device
                    )

                    # Store the model in the loaded_gpus map
                    loaded_gpus[gpu_idx] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "type": "hf"
                    }
                    print(
                        f"Successfully loaded HF model {model_path} on GPU {gpu_idx}")
                except Exception as e:
                    print(f"Error loading HF model {model_path}: {e}")
                    raise e
            else:
                try:
                    # Set CUDA_VISIBLE_DEVICES to restrict which GPU is used
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    print(
                        f"Setting CUDA_VISIBLE_DEVICES={gpu_idx} for model {model_path}")
                    # Use vLLM for non-vector models
                    loaded_gpus[gpu_idx] = {
                        "model": LLM(
                            model=model_path,
                            tensor_parallel_size=1,
                            trust_remote_code=True,
                            dtype="bfloat16",
                            max_model_len=8192 if "gemma" in model_path else 16384  # Set max sequence length
                        ),
                        "type": "vllm"
                    }
                    print(
                        f"Successfully loaded vLLM model {model_path} on GPU {gpu_idx}")
                except Exception as e:
                    print(f"Error loading vLLM model {model_path}: {e}")
                    raise e

        # Assign the loaded model to this agent
        if gpu_idx in loaded_gpus:
            models[model_key] = loaded_gpus[gpu_idx]
            print(f"Assigning model on GPU {gpu_idx} to agent {model_key}")

    # Print summary of model sharing
    gpu_to_agents = {}
    for i in range(1, args.num_agents + 1):
        gpu_idx = getattr(args, f"gpu_{i}")
        if gpu_idx not in gpu_to_agents:
            gpu_to_agents[gpu_idx] = []
        gpu_to_agents[gpu_idx].append(f"model_{i}")

    for gpu_idx, agent_list in gpu_to_agents.items():
        if len(agent_list) > 1:
            print(f"GPU {gpu_idx} is shared by agents: {', '.join(agent_list)}")

    return models

# Function to generate responses using direct Hugging Face model


async def generate_hf_direct(model_data, prompt, args):
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # Set up generation parameters
    max_tokens = 2048 if args.dataset in [
        "math500", "collegemath", "aime2024", "aime2025"] else 1024

    try:
        # Run generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def generate():
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move all tensors to the model's device
            for k, v in inputs.items():
                inputs[k] = v.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_tokens,
                    temperature=args.temp,
                    top_p=0.9,
                    do_sample=True if args.temp > 0 else False,
                    pad_token_id=tokenizer.eos_token_id,  # Use eos_token as pad_token
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response

        return await loop.run_in_executor(None, generate)
    except Exception as e:
        print(f"Error during HF generation: {e}")
        return f"Error generating response: {str(e)}"

# Function to generate responses using direct vLLM instance


async def generate_vllm_direct(model_data, prompt, args):
    model = model_data["model"]

    # Set up sampling parameters
    max_tokens = 2048 if args.dataset in [
        "math500", "collegemath", "aime2024", "aime2025"] else 1024

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    try:
        # Run generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, lambda: model.generate([prompt], sampling_params))
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"Error during vLLM generation: {e}")
        return f"Error generating response: {str(e)}"


async def generate_answer_async(session, model_key, model_name, prompt, args, models=None):
    if args.use_server:
        headers = {"Content-Type": "application/json"}
        # Adjust max_tokens based on dataset
        max_tokens = 2048 if args.dataset in [
            "math500", "collegemath", "aime2024"] else 1024

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": args.temp,
            "top_p": 0.9,
            "n": 1
        }
        try:
            async with session.post(VLLM_API_URLS[model_key], json=payload, headers=headers) as response:
                response_json = await response.json()
                return response_json["choices"][0]["text"].strip()
        except Exception as e:
            print(f"vLLM request failed for {model_name}: {e}")
            return "Error generating response."
    else:
        # Use direct model generation based on model type
        try:
            if models[model_key]["type"] == "hf":
                return await generate_hf_direct(models[model_key], prompt, args)
            else:  # vllm type
                return await generate_vllm_direct(models[model_key], prompt, args)
        except Exception as e:
            print(f"Model generation failed for {model_name}: {e}")
            return "Error generating response."


def determine_available_gpus(args):
    """
    Determine the number of unique GPUs available from the arguments.
    """
    gpu_indices = set()
    for i in range(1, args.num_agents + 1):
        gpu_idx = getattr(args, f"gpu_{i}")
        if gpu_idx is not None:
            gpu_indices.add(gpu_idx)

    # If no GPUs are specified or if using server mode, use num_agents as default
    if not gpu_indices or args.use_server:
        return args.num_agents

    return len(gpu_indices)


async def generate_responses_concurrent(session, agent_contexts, model_names, args, debate, current_responses=None, question=None, models=None):
    # Determine available GPUs
    num_gpus = determine_available_gpus(args)

    # Get all model keys
    all_model_keys = list(agent_contexts.keys())
    responses = [None] * len(all_model_keys)

    # Create GPU-based batching - group agents by their GPU assignment
    gpu_to_agents = {}
    for i, model_key in enumerate(all_model_keys):
        agent_num = int(model_key.split('_')[1])
        gpu_idx = getattr(args, f"gpu_{agent_num}")

        if gpu_idx not in gpu_to_agents:
            gpu_to_agents[gpu_idx] = []
        gpu_to_agents[gpu_idx].append(model_key)

    print(f"GPU assignments: {gpu_to_agents}")

    # Create tasks for each GPU group - processing all GPUs concurrently
    all_tasks = []

    # Process each GPU concurrently
    async def process_gpu_agents(gpu_idx, gpu_agents):
        gpu_results = []
        # Process agents within this GPU sequentially
        for agent_idx, model_key in enumerate(gpu_agents):
            if len(gpu_agents) > 1:
                print(
                    f"Processing agent {model_key} ({agent_idx+1}/{len(gpu_agents)}) on GPU {gpu_idx}")

            if debate != 0:
                # Get other models' responses
                other_opinions = "\n\n".join([
                    f"Agent {key[-1]}'s solution: \n```{response}```"
                    for key, response in current_responses.items()
                    if key != model_key
                ])

                debate_context = (
                    f"\n\nLet's revisit the original question:\n{question}\n"
                    f"You are now in round {debate} of the debate."
                )

                if args.add_self_response:
                    # Include model's own previous response
                    shared_opinions = (
                        "\n\nHere is your previous solution to the question:"
                        f"\nYour previous solution: \n```{current_responses[model_key]}```"
                        f"\n\nThese are the solutions from the other agents:\n{other_opinions}"
                    )
                    opinion_instruction = (
                        "\n\nListen carefully to other agents' solutions and update your solution if necessary. Then, "
                        "provide a reasoning on how your thought process has changed since your previous solution and then provide your final answer."
                    )
                else:
                    # Original behavior - only other agents' responses
                    shared_opinions = "\n\nThese are the solutions from other agents:\n" + other_opinions
                    opinion_instruction = (
                        "\n\nUsing the solutions from other agents as additional information, "
                        "can you provide your answer to the math problem? "
                        "Please update and respond to other agents."
                    )

                # Format the boxed answer requirement based on dataset
                if args.dataset == "math500":
                    answer_format = "\nYour final answer should be in the form \\boxed{answer}, at the end of your response. Please use latex format for your answer (e.g. \\frac{1}{2} over 1/2)"
                elif args.dataset == "collegemath":
                    answer_format = "\nYour final answer should be in the form \\boxed{answer}, at the end of your response. Please use latex format for your answer (e.g. \\frac{1}{2} over 1/2)"
                elif args.dataset == "aime2024":
                    answer_format = "\nYour final answer should be a single integer, in the form \\boxed{answer}, at the end of your response."
                else:  # gsm8k
                    answer_format = "\nYour final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response."

                opinion_instruction += answer_format

                prompt = (
                    debate_context +
                    shared_opinions +
                    opinion_instruction
                )
                formatted_prompt = prompt_formatting(
                    prompt, args.cot, prompt_dict, model_names[model_key])
            else:
                formatted_prompt = agent_contexts[model_key][-1]["content"]

            try:
                # Generate response for this agent
                result = await generate_answer_async(session, model_key, model_names[model_key], formatted_prompt, args, models)
                gpu_results.append((model_key, result))

            except Exception as e:
                error_msg = f"Error from {model_key}: {str(e)}"
                print(error_msg)
                gpu_results.append(
                    (model_key, f"Error generating response: {str(e)}"))

        return gpu_results

    # Create tasks for all GPUs to run in parallel
    gpu_tasks = []
    for gpu_idx, gpu_agents in gpu_to_agents.items():
        if len(gpu_to_agents) > 1:
            print(
                f"Preparing GPU {gpu_idx} with agents: {[key.split('_')[1] for key in gpu_agents]}")
        gpu_tasks.append(process_gpu_agents(gpu_idx, gpu_agents))

    # Run all GPU tasks concurrently
    try:
        all_gpu_results = await asyncio.gather(*gpu_tasks)

        # Flatten results and store in the responses array
        for gpu_results in all_gpu_results:
            for model_key, result in gpu_results:
                batch_index = all_model_keys.index(model_key)
                responses[batch_index] = result
    except Exception as e:
        print(f"Error during parallel GPU processing: {e}")
        # In case of error, set all remaining responses to error
        for i, response in enumerate(responses):
            if response is None:
                responses[i] = f"Error during parallel GPU processing: {str(e)}"

    return responses


def get_model_type(args):
    model_families = []
    for i in range(1, args.num_agents + 1):
        model_path = getattr(args, f"model_{i}")
        model_name = model_path.split("/")[-1]
        model_families.append(model_name)

    # Check if all models are the same
    if all(m == model_families[0] for m in model_families):
        if all(getattr(args, f"model_{i}") == getattr(args, "model_1") for i in range(2, args.num_agents + 1)):
            return "homo" + getattr(args, "model_1").split("/")[-1]
        else:
            return "homo" + "_".join(getattr(args, f"model_{i}").split("/")[-1] for i in range(1, args.num_agents + 1))
    else:
        return "hetero" + "_".join(getattr(args, f"model_{i}").split("/")[-1] for i in range(1, args.num_agents + 1))


def load_prompt(args):
    # Set appropriate prompt path based on dataset
    prompt_path = "src/prompt_debate_template.json"

    with open(prompt_path, "r") as f:
        return json.load(f)


def get_default_output_dir(args):
    if not args.output_dir:
        return f"results/{args.dataset}"  # .upper()}"
    return args.output_dir


def load_dataset_questions(args):
    # Set cache directory if using custom cache
    cache_dir = None
    if args.custom_cache:
        cache_dir = "CUSTOM_CACHE_DIR"  # Replace with your custom cache directory
        print(f"Using custom cache directory for datasets: {cache_dir}")

    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main",
                               split="test", cache_dir=cache_dir)
        return [{"question": item["question"], "answer": item["answer"]} for item in dataset]

    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500",
                               split="test", cache_dir=cache_dir)
        return [{"question": item["problem"], "answer": item["answer"]} for item in dataset]

    elif args.dataset == "collegemath":
        dataset = load_dataset(
            "di-zhang-fdu/College_Math_Test", split="test", cache_dir=cache_dir)
        # Filter by topic if specified
        dataset = dataset.filter(
            lambda x: args.data_topic == x["data_topic"].split(".")[-1])
        return [{"question": item["question"], "answer": item["answer"], "data_topic": item["data_topic"]} for item in dataset]

    elif args.dataset == "aime2024":
        dataset = load_dataset("Maxwell-Jia/AIME_2024",
                               split="train", cache_dir=cache_dir)
        return [{"question": item["Problem"], "answer": item["Answer"]} for item in dataset]

    elif args.dataset == "aime2025":
        dataset = load_dataset("yentinglin/aime_2025",
                               split="train", cache_dir=cache_dir)
        return [{"question": item["problem"], "answer": item["answer"]} for item in dataset]

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_initial_prompts(model_key, model_name, question, args, prompt_dict):
    # Generate different prompts based on agent number and dataset
    i = int(model_key[-1])

    if i % 3 == 0:
        instruction = f"Now here is the given math problem: \n {question}\n Give your reasoning first and then provide your final answer. Your final answer should be {'a single numerical number, ' if args.dataset in ['gsm8k', 'aime2024', 'aime2025'] else ''}in the form \\boxed{{answer}}, at the end of your response."
    elif i % 3 == 1:
        instruction = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be {'a single numerical number, ' if args.dataset in ['gsm8k', 'aime2024', 'aime2025'] else ''}in the form \\boxed{{answer}}, at the end of your response."
    else:
        instruction = f"Solve the following math problem. {question} Explain your reasoning. Your final answer should be {'a single numerical number, ' if args.dataset in ['gsm8k', 'aime2024', 'aime2025'] else ''}in the form \\boxed{{answer}}, at the end of your response."
    if args.dataset in ["math500", "collegemath"]:
        instruction += f" Please use latex format for your answer (e.g., \\frac{1}{2} over 1/2, numbers like pi use \\pi (pi))"

    prompt = prompt_formatting(instruction, args.cot, prompt_dict, model_name)

    return prompt


def get_folder_name(args, model_type):
    folder_name = f"{args.num_agents}_{model_type}_q_{args.num_query}"

    # Add dataset-specific components
    if args.dataset == "collegemath":
        folder_name += f"_topic_{args.data_topic}"

    folder_name += f"_r_{args.round}_temp_{args.temp}"

    if args.cot:
        folder_name += "_cot"
    if args.early_stopping:
        folder_name += '_earlystp'
    if args.add_self_response:
        folder_name += '_addrspns'
    if not args.use_server:
        folder_name += '_direct'
    if args.huggingface:
        folder_name += '_hf'
    if args.custom_cache:
        folder_name += '_custom_cache'

    return folder_name


if __name__ == "__main__":
    args = args_parse()

    # Initialize model names dictionary
    model_names = {
        f"model_{i}": getattr(args, f"model_{i}")
        for i in range(1, args.num_agents + 1)
    }

    # Determine available GPUs
    num_gpus = determine_available_gpus(args)
    if num_gpus < args.num_agents:
        print(
            f"⚠️ Running with {args.num_agents} agents on {num_gpus} GPU(s) - agents will be processed in batches")
    else:
        print(
            f"✅ Running with {args.num_agents} agents on {num_gpus} GPU(s) - all agents will run concurrently")

    # Load models directly if not using server (either vLLM or HF based on huggingface flag)
    models = None

    # Load prompt templates
    prompt_dict = load_prompt(args)

    # Get model type for folder naming
    model_type = get_model_type(args)

    # Set output directory
    args.output_dir = get_default_output_dir(args)
    folder_name = get_folder_name(args, model_type)

    if not args.use_server:
        models = load_models(args)

    if args.use_server:
        print(
            f"✅ Using vLLM API for inference with {args.num_agents} agents on {args.dataset.upper()} dataset.")
    elif args.custom_cache:
        print(
            f"✅ Using custom cache configuration with {args.num_agents} agents on {args.dataset.upper()} dataset.")
    elif args.huggingface:
        print(
            f"✅ Using Hugging Face AutoModel with {args.num_agents} agents on {args.dataset.upper()} dataset.")
    else:
        print(
            f"✅ Using direct vLLM integration with {args.num_agents} agents on {args.dataset.upper()} dataset.")

    # Load questions from dataset
    questions = load_dataset_questions(args)

    async def main():
        generated_description = []
        try:
            async with aiohttp.ClientSession() as session:
                for idx in tqdm(range(len(questions[:args.num_query]))):
                    question_data = questions[idx]
                    question = question_data["question"]
                    answer = question_data["answer"]

                    # Get data_topic for CollegeMath if available
                    data_topic = question_data.get("data_topic", "").split(
                        ".")[-1] if "data_topic" in question_data else None

                    # Initialize agent contexts with appropriate prompts
                    agent_contexts = {}
                    for i in range(1, args.num_agents + 1):
                        model_key = f"model_{i}"
                        prompt = create_initial_prompts(
                            model_key, model_names[model_key], question, args, prompt_dict)
                        agent_contexts[model_key] = [
                            {"model": model_names[model_key], "content": prompt}]

                    print(f"# Question No.{idx+1} starts...")

                    for debate in range(args.round):
                        current_responses = {
                            key: agent_contexts[key][-1]["content"]
                            for key in agent_contexts.keys()
                        } if debate != 0 else None

                        try:
                            responses = await generate_responses_concurrent(
                                session, agent_contexts, model_names, args, debate, current_responses, question, models
                            )

                            answers = []
                            for (model_key, response) in zip(agent_contexts.keys(), responses):
                                agent_contexts[model_key].append(
                                    {"model": model_names[model_key], "content": response})
                                # Only try to parse answers if response doesn't start with "Error"
                                if not response.startswith("Error"):
                                    parsed_answer = parse_answer(
                                        response, args.dataset)
                                    answers.append(parsed_answer)
                                else:
                                    answers.append(None)

                            valid_answers = [
                                ans for ans in answers if ans is not None]
                            if valid_answers and args.early_stopping and all(
                                valid_answers[0] is not None and
                                math_equal(ans, valid_answers[0])
                                for ans in valid_answers
                            ):
                                print(f'Consensus in round {debate}')
                                break
                        except Exception as e:
                            print(f"Error in debate round {debate}: {e}")
                            # Just continue to the next round if there's an error

                    # Save all rounds responses
                    models_response = {
                        f"{model_names[model_key]}{i+1}": [agent_contexts[model_key][j]["content"]
                                                           for j in range(1, len(agent_contexts[model_key]))]
                        for i, model_key in enumerate(agent_contexts.keys())
                    }

                    result = {
                        "question_id": idx,
                        "question": question,
                        "agent_response": models_response,
                        "answer": answer
                    }

                    # Add data_topic for CollegeMath
                    if args.dataset == "collegemath" and data_topic:
                        result["data_topic"] = data_topic

                    generated_description.append(result)
        except Exception as e:
            print(f"Critical error in main loop: {e}")

        return generated_description

    # Run the async main function and get results
    generated_description = asyncio.run(main())

    # Output directory creation and saving results
    try:
        output_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "conversation.json")
        print(f"The result file '{output_file}' is saving...")
        with open(output_file, "w") as f:
            json.dump(generated_description, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")
        # Fallback to saving in current directory
        try:
            with open("conversation_backup.json", "w") as f:
                json.dump(generated_description, f, indent=4)
            print("Saved backup to conversation_backup.json")
        except:
            print("Failed to save results.")

    print("✅ All done!!")
