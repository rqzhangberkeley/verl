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

from verl.utils.reward_score.grader import math_equal


# general reward function for many math datasets.
# RZ: In the training loop, we will call this function in verl.verl.workers.reward_managers.naive.py
# RZ: In NaiveRewardManager.__call__(), it will automatically call self.compute_score function. This function is set to be _default_compute_score in verl.verl.utils.reward_score.__init__.py if we did not override the custom_reward_function.path in the yaml file. Otherwise, it will call the function in the file specified in the yaml file.
# RZ: When defining new reward functions, we need to make sure the function name is compute_score.
# RZ: And the input arguments should be data_source, solution_str, ground_truth, extra_info.

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> bool:
    print("haha I am good.")
    return math_equal(solution_str, f'\\boxed{{{ground_truth}}}', timeout=True)