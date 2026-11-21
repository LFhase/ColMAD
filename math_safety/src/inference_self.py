import json
import requests
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
# Import vLLM
from inference_multi import determine_available_gpus, get_default_output_dir, get_model_type, load_prompt
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--num_agents", type=int, default=1,)
    parser.add_argument("--gpu_1", type=int, default=0,
                        help="GPU index for the model when not using server")
    parser.add_argument("--use_server", action="store_true", default=False,
                        help="Whether to use vLLM server APIs instead of direct integration")
    parser.add_argument("--huggingface", action="store_true", default=False,
                        help="Whether to use HuggingFace models for inference")
    parser.add_argument("--custom_cache", action="store_true", default=False,
                        help="Whether to use custom cache configuration with specified cache directories")
    parser.add_argument("--round", default=4, type=int)
    parser.add_argument("--num_query", default=100, type=int)
    parser.add_argument("--temp", default=0.7, type=float)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--self_consistency",
        default=False,
        action='store_true',
        help="If this is True, apply self-consistency by generating multiple reasoning paths."
    )
    parser.add_argument(
        "--self_refinement",
        default=False,
        action='store_true',
        help="If this is True, apply self-refinement by giving iterative self-feedback."
    )
    parser.add_argument(
        "--num_samples",
        default=5,
        type=int,
        help="Number of reasoning paths to sample for self-consistency."
    )
    parser.add_argument(
        "--refinement_rounds",
        default=16,
        type=int,
        help="Number of refinement rounds for self-refinement."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "math500", "collegemath", "aime2024", "aime2025"],
        help="Dataset to use for inference"
    )
    parser.add_argument(
        "--data_topic",
        type=str,
        default="algebra",
        choices=["algebra", "calculus", "precalculus", "differential_equation",
                 "linear_algebra", "probability", "vector_calculus"],
        help="Topic for CollegeMath dataset (ignored for other datasets)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Base directory to save the result file"
    )
    return parser.parse_args()

# Function to format each model's prompt


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


def refinement_prompt(question, current_response, round, prompt_dict, model_path, dataset):
    # Load the prompt template
    model_list = prompt_dict.keys()
    model_name = model_path.split("/")[-1].split("-")[0].lower()
    for model_key in model_list:
        if model_key in model_name:
            model = model_key
            break

    # Dataset-specific instructions
    if dataset == 'gsm8k':
        instruction = f"Now here is the given math problem: \n {question}\n"
        instruction += f"Here are your response from the previous round:\n"
        instruction += f"Round {round} response : {current_response}\n"
        instruction += f"Please review your previous solution carefully. Provide your feedback and update your solution if needed. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. Round {round + 1}.\n"
    elif dataset == 'aime2024':
        instruction = f"Now here is the given math problem: \n {question}\n"
        instruction += f"Here are your response from the previous round:\n"
        instruction += f"Round {round} response : {current_response}\n"
        instruction += f"Please review your previous solution carefully. Provide your feedback and update your solution if needed. Your final answer should be a single integer, in the form \\boxed{{answer}}, at the end of your response. Round {round + 1}.\n"
    else:  # math500 and collegemath
        instruction = f"Now here is the given math problem: \n {question}\n"
        instruction += f"Here are your response from the previous round:\n"
        instruction += f"Round {round} response : {current_response}\n"
        instruction += f"Please review your previous solution carefully. Provide your feedback and update your solution if needed. Your final answer should be in the form \\boxed{{answer}}, at the end of your response. Please use latex format for your answer (e.g. \\frac{1}{2} over 1/2). Round {round + 1}.\n"

    # Formatting
    formatted_prompt = prompt_dict[model]["prompt_format"].format(
        instruction=instruction)

    return formatted_prompt

# Load vLLM model directly (non-server mode)


def load_vllm_model(args):
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
        node = 0#getattr(args, f"node_{i}")

        # Get current node from SLURM environment
        current_node = 0#os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]
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

# Function to generate using direct vLLM instance


def generate_vllm_direct(model, prompt, args):
    # Set max_tokens based on dataset
    if args.dataset in ["gsm8k"]:
        max_tokens = 1024
    else:  # MATH500 and CollegeMath use 2048
        max_tokens = 2048

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=0.9,
        max_tokens=max_tokens,
        n=args.num_samples if args.self_consistency else 1
    )

    outputs = model.generate([prompt], sampling_params)
    return [choice.text.strip() for choice in outputs[0].outputs]

# Function to call vLLM API and generate response


def generate_answer_vllm(model_key, model_name, prompt, args, VLLM_API_URLS, vllm_model=None):
    if args.use_server:
        headers = {"Content-Type": "application/json"}
        # Set max_tokens based on dataset
        if args.dataset in ["gsm8k", "aime2024"]:
            max_tokens = 1024
        else:  # MATH500 and CollegeMath use 2048
            max_tokens = 2048

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": args.temp,
            "top_p": 0.9,
            "n": args.num_samples if args.self_consistency else 1
        }
        try:
            response = requests.post(
                VLLM_API_URLS[model_key], json=payload, headers=headers)
            response_json = response.json()
            return [choice["text"].strip() for choice in response_json["choices"]]
        except Exception as e:
            print(f"vLLM request failed for {model_name}: {e}")
            return ["Error generating response."]
    else:
        # Use direct vLLM model
        try:
            return generate_vllm_direct(vllm_model, prompt, args)
        except Exception as e:
            print(f"vLLM direct generation failed for {model_name}: {e}")
            return ["Error generating response."]

# Load JSON data


def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_folder_name(args, model_type):
    folder_name = f"{args.num_agents}_{model_type}_q_{args.num_query}"

    # Add dataset-specific components
    if args.dataset == "collegemath":
        folder_name += f"_topic_{args.data_topic}"

    folder_name += f"_r_{args.round}_temp_{args.temp}"

    if args.cot:
        folder_name += "_cot"
    # if args.early_stopping:
    #     folder_name += '_earlystp'
    # if args.add_self_response:
    #     folder_name += '_addrspns'
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
    # print(args.output_dir)
    # exit()
    folder_name = get_folder_name(args, model_type)

    if not args.use_server:
        models = load_vllm_model(args)

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

    # Load dataset based on dataset argument
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        questions = [{"question": item["question"],
                      "answer": item["answer"]} for item in dataset]
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        questions = [{"question": item["problem"], "answer": item["solution"],
                      "level": item["level"]} for item in dataset]
    elif args.dataset == "collegemath":
        dataset = load_dataset("di-zhang-fdu/College_Math_Test", split="test")
        dataset = dataset.filter(
            lambda x: args.data_topic == x["data_topic"].split(".")[-1])
        questions = [{"question": item["question"], "answer": item["answer"],
                      "data_topic": item["data_topic"]} for item in dataset]
    elif args.dataset == "aime2024":
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        questions = [{"question": item["Problem"],
                      "answer": item["Answer"]} for item in dataset]
    elif args.dataset == "aime2025":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
        questions = [{"question": item["problem"],
                      "answer": item["answer"]} for item in dataset]

    generated_description = []
    for idx in tqdm(range(len(questions[:args.num_query]))):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        # Construct prompt based on dataset
        if args.dataset == 'gsm8k':
            instruction = f"Now here is the given math problem: \n {question}\n Give your reasoning first and then provide your final answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
        elif args.dataset in ['aime2024', 'aime2025']:
            instruction = f"Now here is the given math problem: \n {question}\n Give your reasoning first and then provide your final answer. Your final answer should be a single integer, in the form \\boxed{{answer}}, at the end of your response."
        else:
            instruction = f"Now here is the given math problem: \n {question}\n Give your reasoning first and then provide your final answer. Your final answer should be in the form \\boxed{{answer}}, at the end of your response. Please use latex format for your answer (e.g. \\frac{1}{2} over 1/2)"

        agent_contexts = {
            "model_1": [{"model": model_names["model_1"], "content": prompt_formatting(instruction, args.cot, prompt_dict, model_names["model_1"])}]
        }
        print(models)
        # Generate responses
        for model_key in agent_contexts.keys():
            # print(agent_contexts['model_1'][-1])
            if args.self_refinement:
                for round in range(args.refinement_rounds):
                    if round == 0:
                        formatted_prompt = agent_contexts[model_key][-1]["content"]
                    else:
                        current_response = agent_contexts[model_key][-1]["content"]
                        formatted_prompt = refinement_prompt(
                            question, current_response, round, prompt_dict, model_names[model_key], args.dataset)
                    response = generate_answer_vllm(
                        model_key, model_names[model_key], formatted_prompt, args, VLLM_API_URLS, models[model_key]["model"])
                    agent_contexts[model_key].append(
                        {"model": model_names[model_key], "content": response})
            else:
                formatted_prompt = agent_contexts[model_key][-1]["content"]
                response = generate_answer_vllm(
                    model_key, model_names[model_key], formatted_prompt, args, VLLM_API_URLS, models[model_key]["model"])
                agent_contexts[model_key].append(
                    {"model": model_names[model_key], "content": response})

        # Save all responses
        models_response = {
            args.model_1+'1': [agent_contexts["model_1"][i]["content"] for i in range(1, len(agent_contexts["model_1"]))]
        }

        # Create response object with different fields based on dataset
        response_obj = {
            "question_id": idx,
            "question": question,
            "agent_response": models_response,
            "answer": answer
        }

        # Add dataset-specific fields
        if args.dataset == "math500":
            response_obj["level"] = questions[idx]["level"]
        elif args.dataset == "collegemath":
            response_obj["data_topic"] = questions[idx]["data_topic"].split(
                ".")[-1]

        generated_description.append(response_obj)

    # Save results
    output_path = f"{args.output_dir}/{folder_name}"
    os.makedirs(output_path, exist_ok=True)
    print(f"The result file '{output_path}/conversation.json' is saving...")
    with open(f"{output_path}/conversation.json", "w") as f:
        json.dump(generated_description, f, indent=4)

    print("✅ All done!!")
