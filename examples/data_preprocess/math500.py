# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the LIMO dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/math500-base-subset50')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'di-zhang-fdu/MATH500'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}." 
    # RZ: The original instruction is different than that in the OpenR1's codebase.

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')
            question = question_raw + ' ' + instruction_following
            answer = example.pop('answer')
            solution = example.pop('solution')
            subject = example.pop('subject')
            level = example.pop('level')
            unique_id = example.pop('unique_id')

            if args.model_type == 'base':
                prompt = question
            else:
                prompt = [{
                    "role": "user",
                    "content": question
                }]

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "question": question_raw,
                    "answer": solution,
                    "subject": subject,
                    "level": level,
                    "unique_id": unique_id,
                }
            }
            return data
        return process_fn

    model_type = args.model_type

    # RZ: only use 50 examples for testing
    test_dataset = test_dataset.select(range(50)).map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
