<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).
verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

## Setup
#### PPO training flow with batch sizes:
1. The system loads train_batch_size prompts from the dataset
2. For each prompt, it generates rollout.n completions. The real effective batch size becomes train_batch_size * rollout.n, which represents the total number of completions in a training step.
3. These samples are processed in mini-batches of size ppo_mini_batch_size (The number of completions per gradient update)
4. On each GPU, the data is further divided into micro-batches of size ppo_micro_batch_size_per_gpu
5. Gradients are accumulated across these micro-batches before updating the model

#### PPORayTrainer
We implement the RayPPOTrainer, which is a trainer runs on the driver process on a single CPU/GPU node (default is CPU).
The PPORayTrainer include 3 core functions for data preparation, WorkerGroup initialization and PPO training loop.
For more details, see https://verl.readthedocs.io/en/latest/workers/ray_trainer.html.
The training loop is in fit() function.
The computation of PPO micro batches is processed in update_actor and update_critic functions.

#### New dataset
The codebase prepares the gsm8k and lighteval/MATH datasets for PPO training. Datasets should be transferred to parquet format before training. For new datasets, we need to extract the prompt and the solution from the dataset using a customly defined make_map_fn() function. The instruction_following asks the model to put the answer in  \boxed{}. For reference, see examples/data_preprocess/limr.py. 

#### Modify the reward function?
To define a new reward function, just write compute_score() function in a new file in verl/utils/reward_score/ and modify the custom_reward_function.path in verl/trainer/config/ppo_trainer.yaml. If custom_reward_function.path is null, then the RewardManager (see verl/workers/reward_manager/naive.py) will call _default_compute_score() (see verl/utils/reward_score/__init__.py) by default. They implement the reward function for gsm8k (extracting solutions after ####) and lighteval/MATH (using Math-Verify). 

#### Use base model?
We do not need to specially mute the chat template when using base model here since the RlHFDataset class will automatically add the chat template to the prompt (and it will return raw input text for base models). See Line 164 of verl/utils/dataset/rl_dataset.py. It derecctly calls PreTrainedTokenizer.apply_chat_template() method.

## Our Experiments

#### Dataset: MATH500 and DAPO-17k
We mainly use MATH500 and DAPO-17k to train and test the model.

MATH dataset contains 12.5k data, ad 500 of them was sampled in MATH500 as a held-oout test set. We use the remaining 12k for training. See https://github.com/openai/prm800k/tree/main/prm800k/math_splits (this is different from the lighteval/MATH dataset where the training set and test set contain 7.5k and 5k, respectively).

DAPO-17k contains 1.79 million data. We use it for training. In the DAPO's codebase, they use a different prompt than the one used in the original verl codebase. To avoid OOD, we use the original system prompt in the original verl codebase.

#### Compute Prompts Values
In validation time, we compute the value function of each prompt, evaluated at the last token of the prompt. Ideally, this value should be the pass rate of this prompt for the current model (if the value model is well trained). We also sample N=32 responses for each prompt and compute the pass rate.  We find although the prompts' values do not strictly match the pass rate, they have a positive correlation of around 0.6 after about 10-20 RL steps. 