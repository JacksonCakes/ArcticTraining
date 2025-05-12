# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from functools import partial

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process UltraChat dataset configuration.")

    # Add command-line arguments
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="ultrachat",
        help="Path to your UltraChat dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
        help="Number of tensor parallelism splits",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="test",
        help="Output path for the Hugging Face dataset",
    )
    parser.add_argument(
        "--cur_split",
        type=int,
        default=0,
        help="The index of current data generation split",
    )
    parser.add_argument(
        "--total_split",
        type=int,
        default=1,
        help="Total number of data generation splits",
    )
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--gen_prompt_length", type=int, default=128, help="Max tokens to generate")

    # Parse the arguments
    args = parser.parse_args()

    return args


# Load dataset (Hugging Face format)
def load_hf_dataset(dataset):
    result = load_dataset(
        "twinkle-ai/tw-reasoning-instruct-50k",
        split="train",
        num_proc=32,
    )

    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }

    result = result.map(
        partial(
            instruct_format_conversation,
            query_key="input",
            response_key="output",
            source_name="tw-reasoning-instruct-50k",
        )
    )
    return result


# Save responses as a Hugging Face dataset
def save_as_huggingface_dataset(prompts, responses, output_path):
    assert output_path is not None, "Please provide an output_path"
    data = [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, responses)]
    dataset = Dataset.from_dict(
        {
            "prompt": [d["prompt"] for d in data],
            "response": [d["response"] for d in data],
        }
    )
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


def generate(args):
    # Load dataset
    if os.path.dirname(args.output_dataset_path):
        os.makedirs(os.path.dirname(args.output_dataset_path), exist_ok=True)
    f = open(f"{args.output_dataset_path}", "w")

    dataset = load_hf_dataset(args.hf_dataset)
    start = args.cur_split * len(dataset)
    end = (args.cur_split + 1) * len(dataset)
    dataset = dataset.select(range(start, end))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=8192,
        enable_chunked_prefill=True,
        # distributed_executor_backend="ray",
        gpu_memory_utilization=0.96,
    )

    sampling_params = SamplingParams(
        temperature=1,
        top_k=64,
        top_p=0.95,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        detokenize=False,
    )

    def preproc_for_completion(data_sample):
        try:
            prompt_string = tokenizer.apply_chat_template(
                conversation=data_sample["messages"], tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying chat template: {e}. Using raw content if available.")

        tokenized_prompt = tokenizer(
            prompt_string,
            add_special_tokens=False,
        )["input_ids"]

        # ensure the prompt itself isn't too long for the model
        if len(tokenized_prompt) + args.max_tokens > 4096:
            print(
                f"Warning: Prompt length ({len(tokenized_prompt)}) + max_tokens ({args.max_tokens}) exceeds"
                " max_model_len (4096). Truncating prompt."
            )
            # truncate from the left
            allowable_prompt_len = 4096 - args.max_tokens
            if allowable_prompt_len <= 0:
                raise ValueError("max_tokens is too large for max_model_len, no space for prompt.")
            tokenized_prompt = tokenized_prompt[-allowable_prompt_len:]  # Keep the end of the prompt

        return {"processed_prompt_tokens": tokenized_prompt}

    dataset = dataset.map(preproc_for_completion, num_proc=4)

    all_prompts_token_ids = dataset["processed_prompt_tokens"]
    batch_size = 64
    for i in range(0, len(all_prompts_token_ids), batch_size):
        outputs = llm.generate(
            sampling_params=sampling_params, prompt_token_ids=all_prompts_token_ids[i : i + batch_size]
        )
        for o in outputs:
            new_data = {"input": o.prompt_token_ids, "output": o.outputs[0].token_ids}
            f.write(json.dumps(new_data) + "\n")
            f.flush()


def main():

    args = parse_arguments()

    generate(args)


if __name__ == "__main__":
    main()
