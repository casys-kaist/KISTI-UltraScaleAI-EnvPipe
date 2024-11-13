import json
import os
from datasets import load_dataset

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), 'download')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'json')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_dataset(dataset, prompt_key, completion_key, input_key=None):
    prompts = [data[prompt_key] + (data[input_key] if input_key else '') for data in dataset]
    completions = [data[completion_key] for data in dataset]
    return list(zip(prompts, completions))


def load_gsm8k():
    dataset = load_dataset('gsm8k', 'main', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'question', 'answer')


def load_humaneval():
    dataset = load_dataset('openai_humaneval', cache_dir=DOWNLOAD_DIR)['test']
    return process_dataset(dataset, 'prompt', 'canonical_solution')


def load_alpaca():
    dataset = load_dataset('tatsu-lab/alpaca', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'instruction', 'output', input_key='input')


def load_apps():
    dataset = load_dataset('codeparrot/apps', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'question', 'solutions')


def load_dialogue():
    dataset = load_dataset('facebook/empathetic_dialogues', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'prompt', 'utterance')


def load_chatbot():
    dataset = load_dataset('alespalla/chatbot_instruction_prompts', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'prompt', 'response')


def load_finance():
    dataset = load_dataset('gbharti/finance-alpaca', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'instruction', 'output')


def sample_requests(dataset_name: str):
    if dataset_name == 'gsm8k':
        return load_gsm8k()
    elif dataset_name == 'humaneval':
        return load_humaneval()
    elif dataset_name == 'alpaca':
        return load_alpaca()
    elif dataset_name == 'apps':
        return load_apps()
    elif dataset_name == 'dialogue':
        return load_dialogue()
    elif dataset_name == 'chatbot':
        return load_chatbot()
    elif dataset_name == 'finance':
        return load_finance()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    import argparse

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Download and process entire dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "gsm8k", "humaneval", "alpaca", "apps", "dialogue", "chatbot", "finance"
        ],
        help="Name of the dataset to download and process."
    )

    args = parser.parse_args()

    # Load the specified dataset
    dataset = sample_requests(args.dataset)

    # Prepare output file path
    output_path = os.path.join(OUTPUT_DIR, f"{args.dataset}.json")

    # Save each prompt-completion pair as a line in the JSON file
    with open(output_path, 'w') as f:
        for prompt, output in dataset:
            json.dump({"prompt": prompt, "output": output}, f)
            f.write('\n')

    print(f"Downloaded and saved dataset '{args.dataset}' with {len(dataset)} samples to {output_path}.")


if __name__ == '__main__':
    main()
