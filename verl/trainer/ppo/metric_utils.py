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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto
import wandb


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def compute_pass_rate_metrics(prompt_stats, prompt_value_avg=None):
    """Compute pass rate metrics and create visualization for wandb logging.
    
    Args:
        prompt_stats (dict): Dictionary containing prompt statistics with pass rates
        prompt_value_avg (Optional[dict]): Dictionary containing prompt value averages
        
    Returns:
        tuple: (metric_dict, wandb_plot) where metric_dict contains the metrics and wandb_plot is the bar plot
    """
    metric_dict = {}
    pass_rates = [stats['pass_rate'] for stats in prompt_stats.values()]
    
    # Calculate basic pass rate statistics
    metric_dict['val/pass_rate/avg'] = np.mean(pass_rates)
    metric_dict['val/pass_rate/median'] = np.median(pass_rates)
    
    # Calculate percentages for different pass rate buckets
    buckets = {
        '0%': sum(1 for rate in pass_rates if rate == 0.0),
        '0-20%': sum(1 for rate in pass_rates if 0.0 < rate < 0.2),
        '20-40%': sum(1 for rate in pass_rates if 0.2 <= rate < 0.4),
        '40-60%': sum(1 for rate in pass_rates if 0.4 <= rate <= 0.6),
        '60-80%': sum(1 for rate in pass_rates if 0.6 < rate < 0.8),
        '80-100%': sum(1 for rate in pass_rates if 0.8 <= rate < 1.0),
        '100%': sum(1 for rate in pass_rates if rate == 1.0)
    }
    
    total_prompts = len(pass_rates)
    for bucket, count in buckets.items():
        metric_dict[f'val/pass_rate/bucket_{bucket}'] = count / total_prompts
    
    # Create wandb bar plot
    data = [[bucket, (count/total_prompts)*100] for bucket, count in buckets.items()]
    table = wandb.Table(data=data, columns=['Pass Rate Bucket', 'Percentage of Prompts'])
    wandb_plot = wandb.plot.bar(
        table, 
        'Pass Rate Bucket',
        'Percentage of Prompts',
        title='Distribution of Pass Rates'
    )
    
    # If prompt values are provided, compute last token metrics for different pass rate ranges
    if prompt_value_avg is not None:
        last_values = [stats.get('value_last') for stats in prompt_stats.values()]
        
        # Compute metrics for different pass rate ranges using last token values only
        ranges = [
            (1.0, 1.0, 'perfect'),
            (0.0, 0.0, 'failed'),
            (0.8, 1.0, 'high_pass'),
            (0.0, 0.2, 'low_pass'),
            (0.4, 0.6, 'medium_pass')
        ]
        
        for min_rate, max_rate, label in ranges:
            values = [
                stats.get('value_last') 
                for stats in prompt_stats.values() 
                if min_rate <= stats['pass_rate'] <= max_rate
            ]
            metric_dict[f'val/prompt_value/{label}_prompts_last'] = np.mean(values) if values else -1
        
        # Compute correlation between pass rates and last token values
        value_last_correlation = np.corrcoef(pass_rates, last_values)[0, 1]
        metric_dict['val/prompt_value/last_correlation_with_pass'] = value_last_correlation
    
    return metric_dict, wandb_plot
