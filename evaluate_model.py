import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
from datetime import datetime

def get_model_type(model_name):
    if model_name in ['Qwen/Qwen2.5-Math-1.5B','Qwen/Qwen2.5-Math-7B']:
        return 'base'
    elif model_name in ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B']:
        return 'distill'
    else:
        raise ValueError(f"Model {model_name} not supported.")

def get_scores(ds,
            outputs, 
            save_file_name=None,
            answer_key=None,
            question_key=None,
            response_extractor=None,
            response_comparator=None
            ):

    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs): # output is a list of vllm.generate.GenerateResult objects. There can be multiple outputs per input in the output list.
        gold = response_extractor(input[answer_key])
        prediction = [
            response_extractor(resp.text)
            for resp in output.outputs
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                question_key: input[question_key],
                answer_key: input[answer_key],
                "responses": [resp.text for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
                "accuracy": [response_comparator(gold, pred) for pred in prediction],
            }
        )
    if save_file_name is not None: # save.
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    pass_at_1 = sum([any([response_comparator(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions) # pass@1
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))

    # pass@k
    for i in range(k):
        pass_at_i = sum([any([response_comparator(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_i = sum([response_comparator(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_k_list.append(acc_at_i)
        pass_at_k_list.append(pass_at_i)
        print(
            f"Pass @ {i+1}: {pass_at_i}"
        )

    # determine the most common answer. Compute pass@1(majority)
    def get_most_common(solns):
        soln_counts = {}
        for soln in solns:
            if soln is None:
                continue
            added = False
            for other_solns in solns:
                if response_comparator(soln, other_solns):
                    added = True
                    soln_counts[soln] = soln_counts.get(soln, 0) + 1
            if not added:
                soln_counts[soln] = 1
        if len(soln_counts) == 0:
            return None
        return max(soln_counts, key=soln_counts.get)
    
    predictions_maj = [get_most_common(p) for p in predictions]
    all_preds = sum([[response_comparator(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
    avg_pass_rate = sum(all_preds) / len(all_preds)
    pass_at_n = sum([response_comparator(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
    print(
        f"Pass @ 1(with majority): {pass_at_n}"
    )
    
    return {
        'pass@1': pass_at_1,
        'pass@1(majority)': sum([response_comparator(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
        'average_pass_rate': avg_pass_rate,
        'std_pass_rate': np.std(acc_at_k_list),
        'acc@k': acc_at_k_list,
        'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }

def evaluate_model(model_name,
                    dataset,
                    dataset_name,
                    dataset_split,
                    answer_key,
                    question_key,
                    response_extractor,
                    response_comparator,
                    tok_limit,
                    test_n,
                    max_test_samples,
                    test_temperature,
                    example_prompt=None,
                    example_solution=None,
                    n_gpus=1):
    test_prompts = []
    model = LLM(model_name, 
                tokenizer=f'{model_name}',
                # tokenizer='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                gpu_memory_utilization=0.9, 
                tensor_parallel_size=n_gpus)
    print(f"Model dtype: {model.llm_engine.model_config.dtype}")
    test_ds = dataset[dataset_split].shuffle(seed=0).select(range(min(max_test_samples, len(dataset[dataset_split]))))
    
    for x in test_ds:
        if get_model_type(model_name) == 'base':
            prompt_tokens = "Please reason step by step, and put your final answer within \\boxed{{}}. Question: " + x[question_key]
            prompt_tokens = model.llm_engine.tokenizer.tokenizer.encode(prompt_tokens, add_special_tokens=False)
            test_prompts.append(prompt_tokens)
        else:
            raise NotImplementedError
    
    sampling_params = SamplingParams(
        temperature=test_temperature,
        max_tokens=tok_limit,
        n=test_n
    )
    if example_prompt is None:
        save_file_name = f"outputs/{dataset_name.replace('/', '_')}_results_{model_name.replace('/', '_')}_{tok_limit}.json"
    elif isinstance(example_prompt,str):
        save_file_name = f"outputs/{dataset_name.replace('/', '_')}_results_{model_name.replace('/', '_')}_{tok_limit}_1shot.json"
    else:
        raise NotImplementedError
    
    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    
    print("Generating test outputs...")
    print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))

    start_time = time.time()
    test_outputs = model.generate(prompt_token_ids=test_prompts, 
                                sampling_params=sampling_params, 
                                use_tqdm=True) # generate outputs
    end_time = time.time()
    test_scores = get_scores(test_ds, 
                            test_outputs, 
                            save_file_name,
                            answer_key,
                            question_key,
                            response_extractor,
                            response_comparator) # get scores
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}


# This script evaluates a model on a dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', type=str, nargs='+', default=[])
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--scale', type=str, default='auto')
    parser.add_argument('--tok_limit', type=int, default=32768)
    parser.add_argument('--n_gpus',type=int,default=1)
    parser.add_argument('--use_example_prompt', type=bool, default=False)
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ["VLLM_ATTENTION_ENGINE"] = "flash-v2"

    dataset_name = args.dataset
    model_paths = args.model_paths
    scale = args.scale
    tok_limit = args.tok_limit
    dataset_name = args.dataset
    n_gpus=args.n_gpus
    model_scales = []
    print(f"\n\nmodel_paths = {model_paths}\n\n") 
    print(f"\n\nscale = {scale}\n\n")
    print(f"\n\nn_gpus = {n_gpus}\n\n")

    if scale == 'auto':
        for model_path in model_paths:
            if not any(size in model_path for size in ['1.5B', '7B', '14B', '32B']):
                raise ValueError(f"Model path {model_path} must contain a param size, e.g. '1.5B' or '7B' or '14B' or '32B'")
            for size in ['1.5B', '7B', '14B', '32B']:
                if size in model_path:
                    model_scales.append(size)
                    continue
    else:
        model_scales = [scale]

    print("Dataset:", dataset_name)
    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    print(f"\n\nQUESTION_KEY = {QUESTION_KEY}\n\n")
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    print(f"\n\nANSWER_KEY = {ANSWER_KEY}\n\n")
    eq = RESPONSE_COMPARATOR[dataset_name]

    if dataset_name == 'datasets/converted_aime_dataset':
        dataset = load_from_disk(dataset_name)
        TEST_N = 10
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 100
        DATASET_SPLIT = 'test'
    elif dataset_name == 'di-zhang-fdu/MATH500':
        dataset = load_dataset(dataset_name)
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 500
        DATASET_SPLIT = 'test'
    elif dataset_name == 'openai/gsm8k': # RZ: Let's first try gsm8k
        dataset = load_dataset(dataset_name, 'main')
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 1319 # RZ: size of gsm8k eval.
        DATASET_SPLIT = 'test'
    elif dataset_name == 'GAIR/LIMO':
        dataset = load_dataset(dataset_name)
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 817
        DATASET_SPLIT = 'train'
    elif dataset_name == 'GAIR/LIMR':
        dataset = load_dataset(dataset_name)
        TEST_N = 8
        MAX_TOKENS = tok_limit
        TEST_TEMPERATURE = 0.6
        MAX_TEST_SAMPLES = 1389
        DATASET_SPLIT = 'train'
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if args.use_example_prompt:
        example_prompt_dir = './example_prompt.json'
        with open(example_prompt_dir, 'r') as f:
            example = json.load(f)
            example_prompt = example['prompt']
            example_solution = example['solution']
    else:
        example_prompt = None
        example_solution = None
    

    for model_path, scale in zip(model_paths, model_scales):
        results = {}
        print("Found model_path:", model_path)
        print("This is not a checkpoint, will evaluate directly...")
        scores = evaluate_model(model_path, 
                                dataset, 
                                dataset_name,
                                DATASET_SPLIT,
                                ANSWER_KEY, 
                                QUESTION_KEY, 
                                RESPONSE_EXTRACTOR[dataset_name],
                                eq, 
                                tok_limit, 
                                TEST_N, 
                                MAX_TEST_SAMPLES, 
                                TEST_TEMPERATURE,
                                example_prompt=example_prompt,
                                example_solution=example_solution,
                                n_gpus=n_gpus
                            )
        results[model_path] = scores

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if example_prompt is None:
            results_file = f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}_{current_time}.json'
        elif isinstance(example_prompt,str):
            results_file = f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}_1shot_singleturn_{current_time}.json'
        else:
            raise NotImplementedError

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Finished evaluating {model_path} on {dataset_name} with {scale}.")


# import os, time
# import json
# from vllm import LLM, SamplingParams
# from datasets import load_from_disk, load_dataset
# from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
# import pandas as pd
# import argparse
# import numpy as np


# # This script evaluates a model on a dataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, default='')
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--scale', type=str, default='1.5B')
# parser.add_argument('--tok_limit', type=int, default=32768)
# args = parser.parse_args()
# os.environ['TOKENIZERS_PARALLELISM'] = "false"

# dataset_name = args.dataset
# model_path = args.model_path
# scale = args.scale
# tok_limit = args.tok_limit
# dataset_name = args.dataset
# results = {}

# print("Dataset:", dataset_name, "\nScale:", scale)

# QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
# ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
# eq = RESPONSE_COMPARATOR[dataset_name]

# if dataset_name == 'datasets/converted_aime_dataset':
#     dataset = load_from_disk(dataset_name)
#     TEST_N = 10
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 100
# elif dataset_name == 'di-zhang-fdu/MATH500':
#     dataset = load_dataset(dataset_name)
#     TEST_N = 3
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 500
# elif dataset_name == 'openai/gsm8k':
#     dataset = load_dataset(dataset_name, 'main')
#     TEST_N = 1
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 1319


# def get_scores(ds, outputs, save_file_name=None):
#     predictions, golds = [], []
#     results = []
#     for input, output in zip(ds, outputs):
#         gold = RESPONSE_EXTRACTOR[dataset_name](input[ANSWER_KEY])
#         prediction = [
#             RESPONSE_EXTRACTOR[dataset_name](resp.text)
#             for resp in output.outputs
#         ]
#         predictions.append(prediction)
#         golds.append(gold)
#         results.append(
#             {
#                 QUESTION_KEY: input[QUESTION_KEY],
#                 ANSWER_KEY: input[ANSWER_KEY],
#                 "responses": [resp.text for resp in output.outputs],
#                 "prediction": prediction,
#                 "gold": gold,
#                 "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
#                 "accuracy": [eq(gold, pred) for pred in prediction],
#             }
#         )
#     if save_file_name is not None:
#         with open(save_file_name, 'w') as f:
#             json.dump(results, f, indent=4)

#     results = pd.DataFrame(results)
#     predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
#     pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
#     pass_at_k_list = []
#     acc_at_k_list = []
#     k = TEST_N
#     print("Average tokens:", sum(tokens) / len(tokens))
#     for i in range(k):
#         pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
#         acc_at_i = sum([eq(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
#         acc_at_k_list.append(acc_at_i)
#         pass_at_k_list.append(pass_at_i)
#         print(
#             f"Pass @ {i+1}: {pass_at_i}"
#         )

#     def get_most_common(solns):
#         soln_counts = {}
#         for soln in solns:
#             if soln is None:
#                 continue
#             added = False
#             for other_solns in solns:
#                 if eq(soln, other_solns):
#                     added = True
#                     soln_counts[soln] = soln_counts.get(soln, 0) + 1
#             if not added:
#                 soln_counts[soln] = 1
#         if len(soln_counts) == 0:
#             return None
#         return max(soln_counts, key=soln_counts.get)
    
#     predictions_maj = [get_most_common(p) for p in predictions]
#     all_preds = sum([[eq(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
#     avg_pass_rate = sum(all_preds) / len(all_preds)
#     pass_at_n = sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
#     print(
#         f"Pass @ 1(with majority): {pass_at_n}"
#     )
    
#     return {
#         'pass@1': pass_at_1,
#         'pass@1(majority)': sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
#         'average_pass_rate': avg_pass_rate,
#         'std_pass_rate': np.std(acc_at_k_list),
#         'acc@k': acc_at_k_list,
#         'pass@k': pass_at_k_list,
#         'avg_tokens': sum(tokens) / len(tokens)
#     }


# def evaluate_model(model_name):
#     test_prompts = []
#     model = LLM(model_name, tokenizer=f'deepseek-ai/DeepSeek-R1-Distill-Qwen-{scale}', gpu_memory_utilization=0.9, tensor_parallel_size=1)    
#     test_ds = dataset['test'].shuffle(seed=1001).select(range(min(MAX_TEST_SAMPLES, len(dataset['test']))))
    
#     for x in test_ds:
#         prompt = [{
#             "role": "user",
#             "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {x[QUESTION_KEY]}",
#         }]
#         prompt_tokens = model.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
#         test_prompts.append(prompt_tokens)
    
#     sampling_params = SamplingParams(
#         temperature=TEST_TEMPERATURE,
#         max_tokens=MAX_TOKENS,
#         n=TEST_N
#     )
#     sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
#     print("Generating test outputs...")
#     print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
#     start_time = time.time()
#     test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
#     end_time = time.time()
#     test_scores = get_scores(test_ds, test_outputs, f"outputs/{dataset_name.replace('/', '_')}_results_{model_path.replace('/', '_')}_{tok_limit}.json")
#     print("Test:", test_scores)
#     time_taken = end_time - start_time
#     print("Time taken:", time_taken)

#     return {'test': test_scores, 'time_taken': time_taken}

# print("Found model_path:", model_path)
# print("This is not a checkpoint, will evaluate directly...")
# scores = evaluate_model(model_path)
# results[model_path] = scores

# with open(f'results/{dataset_name.replace("/", "_")}_results_{model_path.replace("/", "_")}_{tok_limit}.json', 'w') as f:
#     json.dump(results, f, indent=4)