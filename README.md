<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLM</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).
verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.


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

#### Test run
We tried to run PPO on Qwen2.5/0.5B-Instruct model on gsm8k for 15 epochs (based on the default hyperparameters). The validation accuracy rises from almost zero to about 52% after 15 epochs. The initial accuracy is lower than the standard result reported by Qwen (about 50) or the results from the offline evaluation codebase (45-47), because we are using a different prompt (by asking the model to put the answer after ####), but the model learns to follow this instruction quickly. According to verl's report, their trained 0.5B-Instruct model achieves 56.7 on validation set. We have not reproduced this result yet.

We also train 1.5B-Instruct model on Lighteval/MATH dataset via PPO for 3 epochs. For MATH dataset, we need math-verify package, which need to manually install antlr4-python3-runtime==4.9.3 (I am not sure why, but it seems the default installation pipeline does not work). We set the max_prompt_length to 1024 and we filter the overlong prompts. The training reward rises from around 48 to 59, but the validation accuracy rises from 55.6 to 56.1, which is not good. We train the model for longer epochs (15 epochs) and find the validation accuracy rises from 55.8 to 57 and then drops to 54. 

We also train a 7B-Instruct model on Lighteval/MATH with global batch size = 1024 (7 steps per epoch) for 3 epochs and it takes around 1 hour on 8 GPUs. 