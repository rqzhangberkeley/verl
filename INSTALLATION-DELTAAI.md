#### Installation on DeltaAI

1. To login the DeltaAI cluster, please regiister an account and then setup the 2 factor authentication.

2. You need to install miniconda by yourself. You can ask GPT for help. This is regular.

3. Load the modules. These lines are very important. Always run these lines from time to time. When you activate a new conda environment, you need to run these lines before you start to install packages. Whenever you have bugs, try to run these lines.
```
module default
module load cuda/12.6.1
module load gcc/11.4.0
```

4. Request GPU node. It will be better for yoou to install packages in the GPU node. To request an GPU node, run the following command. For the 'accounts', please run 'accounts' command to check your account name. You should request for more CPUs and memory.
```
salloc --gpus=1 --cpus-per-task=200 --mem=400G --partition=ghx4-interactive --account=bdwy-dtai-gh --time=01:00:00
```
After you enter the GPPU node and activate the conda environment, you need to load the modules again (see step 3).

5. Install the nighty version of torch. The concrete version of torch is not important. You can install the latest version. But it must be the nighty version.
```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

6. Install the flash_attn from source. Run the following command in the GPU node.
```
MAX_JOBS=8
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.7.4
pip install -e .
```

7. Install triton from source. Run the following.
```
git clone https://github.com/openai/triton.git
cd triton/python
pip install -e .
```

8. Install vllm from source. After git clone, you should first hardcode the get_vllm_version() function in the setup.py file to make it return the version you want, for example, '0.6.3'. Then, you can run python use_existing_torch.py.
```
pip install setuptools_scm cmake ninja
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Hardcode the setup.py file
python use_existing_torch.py
git checkout v0.6.3
pip install --no-build-isolation -e .
```
Do not forget --no-build-isolation here. It is necessary.

9. Install the other dependencies and verl.
```
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
```