# Reinforcement Learning Algorithms for Language Models

This document provides a detailed explanation of the reinforcement learning (RL) algorithms implemented in the VERL framework for training language models. We will cover Proximal Policy Optimization (PPO), RLOO, Generalized Reward Policy Optimization (GRPO), and REINFORCE++, focusing on their implementation details, objective functions, and key hyperparameters.

## Table of Contents

- [1. Proximal Policy Optimization (PPO)](#1-proximal-policy-optimization-ppo)
  - [1.1 Overview](#11-overview)
  - [1.2 Model Components](#12-model-components)
  - [1.3 Generalized Advantage Estimation (GAE)](#13-generalized-advantage-estimation-gae)
  - [1.4 Objective Functions](#14-objective-functions)
  - [1.5 Training Process](#15-training-process)
  - [1.6 Key Hyperparameters](#16-key-hyperparameters)
  - [1.7 Hyperparameter Mapping in Codebase](#17-hyperparameter-mapping-in-codebase)
  - [1.8 PPO Algorithm Box](#18-ppo-algorithm-box)
- [2. Leave-One-Out Reinforcement Learning (RLOO)](#2-leave-one-out-reinforcement-learning-rloo)
  - [2.1 Overview](#21-overview)
  - [2.2 Objective Functions](#22-objective-functions)
  - [2.3 Implementation Details](#23-implementation-details)
  - [2.4 Key Hyperparameters](#24-key-hyperparameters)
  - [2.5 RLOO Algorithm Box](#25-rloo-algorithm-box)
- [3. Generalized Reward Policy Optimization (GRPO)](#3-generalized-reward-policy-optimization-grpo)
  - [3.1 Overview](#31-overview)
  - [3.2 Objective Functions](#32-objective-functions)
  - [3.3 Implementation Details](#33-implementation-details)
  - [3.4 Key Hyperparameters](#34-key-hyperparameters)
  - [3.5 GRPO Algorithm Box](#35-grpo-algorithm-box)
- [4. REINFORCE++](#4-reinforce)
  - [4.1 Overview](#41-overview)
  - [4.2 Objective Functions](#42-objective-functions)
  - [4.3 Implementation Details](#43-implementation-details)
  - [4.4 Key Hyperparameters](#44-key-hyperparameters)
  - [4.5 REINFORCE++ Algorithm Box](#45-reinforce-algorithm-box)
- [5. Rule-Based RL Training](#5-rule-based-rl-training)
  - [5.1 Overview](#51-overview)
  - [5.2 Binary Reward Function](#52-binary-reward-function)
  - [5.3 Integration with RL Algorithms](#53-integration-with-rl-algorithms)
  - [5.4 Rule-Based RL for Mathematical Problem-Solving](#54-rule-based-rl-for-mathematical-problem-solving)

## 1. Proximal Policy Optimization (PPO)

### 1.1 Overview

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning that aims to strike a balance between ease of implementation, sample complexity, and ease of tuning. In the context of language models, PPO is used to fine-tune models based on human preferences or reward signals, thereby aligning the model's outputs with desired behaviors.

### 1.2 Model Components

The PPO implementation in the VERL framework includes several key components:

1. **Actor (Policy) Model**: The language model being trained to generate text. It outputs logits over the vocabulary for each token position.

2. **Critic (Value) Model**: Estimates the value function, predicting the expected future rewards for a given state. Used to compute advantages.

3. **Rollout Model**: Used to generate text responses during the data collection phase.

4. **Reference Model**: A frozen copy of the initial policy model used to calculate KL divergence to prevent the policy from deviating too far from the original model.

5. **Reward Model**: Provides the reward signal that guides the optimization. Can be a separate model or a function.

### 1.3 Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation (GAE) is a method used in PPO to estimate the advantage function. The advantage function indicates how much better an action is compared to the average action in a given state.

**Mathematical Formulation:**

The GAE for a given time step $t$ is calculated as:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t$ is the TD error:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\gamma$ is the discount factor, $\lambda$ is the GAE parameter, $r_t$ is the reward at time $t$, and $V(s_t)$ is the value function at state $s_t$.

GAE is implemented in the code as follows (from `verl/trainer/ppo/core_algos.py`):

```python
def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, 
                                 eos_mask: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    
    Parameters:
        token_level_rewards: Rewards for each token in each sequence. Shape: [batch_size, sequence_length]
        values: Value function estimates for each token position. Shape: [batch_size, sequence_length+1]
          (includes a final value estimate for bootstrapping)
        eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
        gamma: Discount factor for future rewards. Shape: [] (scalar)
        lam: GAE lambda parameter for bias-variance tradeoff. Shape: [] (scalar)
        
    Returns:
        advantages: Calculated advantages. Shape: [batch_size, sequence_length]
        returns: Calculated returns. Shape: [batch_size, sequence_length]
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns
```

Key parameters in GAE:
- **gamma**: The discount factor for future rewards (typically close to 1, e.g., 0.99)
- **lam**: The lambda parameter that controls the trade-off between bias and variance (typically around 0.95)

GAE computes advantages by calculating the temporal-difference (TD) error at each time step and then discounting these errors with both gamma and lambda parameters.

### 1.4 Objective Functions

PPO has several objective functions that are combined during training:

1. **Policy Loss**:
   
   The mathematical expression for the PPO policy loss is:

   $$L^{CLIP}(\theta) = \hat{E}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

   where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies, $\hat{A}_t$ is the estimated advantage, and $\epsilon$ is the clip range.

   ```python
   # From verl/trainer/ppo/core_algos.py
   def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
       """
       Compute PPO's clipped policy gradient loss.
       
       Parameters:
           old_log_prob: Log probabilities from the old policy. Shape: [batch_size, sequence_length]
           log_prob: Log probabilities from the current policy. Shape: [batch_size, sequence_length]
           advantages: Advantage estimates. Shape: [batch_size, sequence_length]
           eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
           cliprange: PPO clipping parameter for limiting policy updates. Shape: [] (scalar)
           
       Returns:
           pg_loss: Calculated policy gradient loss. Shape: [] (scalar)
       """
       # Compute the ratio of new policy to old policy
       ratio = torch.exp(log_prob - old_log_prob)
       
       # Clipped surrogate objective
       pg_loss1 = -advantages * ratio
       pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
       pg_loss = torch.maximum(pg_loss1, pg_loss2)
       
       # Apply response mask and compute mean
       pg_loss = verl_F.masked_mean(pg_loss, eos_mask)
       
       return pg_loss
   ```

   The policy loss uses a clipped surrogate objective to prevent too large policy updates, helping with stability.

2. **Value Loss**:
   
   The mathematical expression for the PPO value loss is:

   $$L^{VF}(\theta) = \hat{E}_t \left[ \max\left((V_\theta(s_t) - R_t)^2, (V_{\theta_{clip}}(s_t) - R_t)^2 \right) \right]$$

   where $V_{\theta_{clip}}(s_t) = V_{\theta_{old}}(s_t) + \text{clip}(V_\theta(s_t) - V_{\theta_{old}}(s_t), -\epsilon_v, \epsilon_v)$, $R_t$ is the return, and $\epsilon_v$ is the value clip range.

   ```python
   # From verl/trainer/ppo/core_algos.py
   def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
       """
       Compute PPO's clipped value function loss.
       
       Parameters:
           vpreds: Value predictions from the current critic model. Shape: [batch_size, sequence_length]
           returns: Target returns for value function. Shape: [batch_size, sequence_length]
           values: Value predictions from the old critic model. Shape: [batch_size, sequence_length]
           eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
           cliprange_value: Clipping parameter for value function updates. Shape: [] (scalar)
           
       Returns:
           vf_loss: Calculated value function loss. Shape: [] (scalar)
       """
       # Clipped value function objective
       vpredclipped = values + torch.clamp(vpreds - values, -cliprange_value, cliprange_value)
       vf_losses1 = (vpreds - returns) ** 2
       vf_losses2 = (vpredclipped - returns) ** 2
       vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2)
       
       # Apply response mask and compute mean
       vf_loss = verl_F.masked_mean(vf_loss, eos_mask)
       
       return vf_loss
   ```

   The value loss uses a clipped objective similar to the policy loss to ensure stable updates to the value function.

3. **Entropy Loss**:
   
   The mathematical expression for the entropy loss is:

   $$L^{ENT}(\theta) = -\hat{E}_t \left[ \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]$$

   where $\pi_\theta(a|s_t)$ is the probability of taking action $a$ in state $s_t$ under policy $\pi_\theta$.

   ```python
   # From verl/trainer/ppo/core_algos.py
   def compute_entropy_loss(logits, eos_mask):
       """
       Compute policy entropy loss to encourage exploration.
       
       Parameters:
           logits: Unnormalized log probabilities from policy model. Shape: [batch_size, sequence_length, vocab_size]
           eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
           
       Returns:
           entropy: Calculated entropy loss. Shape: [] (scalar)
       """
       # Compute entropy for each token position
       entropy_per_token = verl_F.entropy_from_logits(logits)
       
       # Apply response mask and compute mean
       entropy = verl_F.masked_mean(entropy_per_token, eos_mask)
       
       return entropy
   ```

   The entropy loss encourages exploration by penalizing policies that are too deterministic.

4. **KL Penalty**:
   
   The mathematical expression for the KL penalty is:

   $$L^{KL}(\theta) = \beta \cdot \text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

   where $\beta$ is the KL coefficient, and $\text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$ is the KL divergence between the reference policy and the current policy.

   ```python
   # From verl/trainer/ppo/core_algos.py
   def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
       """
       Compute KL divergence penalty between current policy and reference policy.
       
       Parameters:
           logprob: Log probabilities from the current policy. Shape: [batch_size, sequence_length]
           ref_logprob: Log probabilities from the reference (initial) policy. Shape: [batch_size, sequence_length]
           kl_penalty: Type of KL penalty to use ('kl', 'abs', or 'mse'). Shape: [] (string)
           
       Returns:
           penalty: Calculated KL penalty. Shape: [batch_size, sequence_length]
       """
       if kl_penalty == 'kl':
           return ref_logprob - logprob
       elif kl_penalty == 'abs':
           return torch.abs(ref_logprob - logprob)
       elif kl_penalty == 'mse':
           return 0.5 * (ref_logprob - logprob) ** 2
       else:
           raise NotImplementedError
   ```

   The KL penalty prevents the policy from deviating too far from the reference policy, controlled by an adaptive or fixed coefficient.

5. **Total Loss**:

   The overall PPO objective combines these losses:

   $$L^{TOTAL}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 L^{ENT}(\theta) - c_3 L^{KL}(\theta)$$

   where $c_1$, $c_2$, and $c_3$ are coefficients to weight the different loss components.

### 1.5 Training Process

The PPO training process in VERL follows these steps:

1. **Data Collection**: Generate responses using the current policy model.
2. **Reward Calculation**: Compute rewards for each response.
3. **Advantage Estimation**: Use GAE to compute advantages.
4. **Policy Update**: Update the policy model using the clipped surrogate objective.
5. **Value Function Update**: Update the value function to better predict returns.
6. **KL Coefficient Update**: Adjust the KL penalty coefficient to maintain the desired KL divergence.

### 1.6 Key Hyperparameters

- **gamma** (discount factor): Typically around 0.99, controls the importance of future rewards.
- **lam** (GAE parameter): Typically around 0.95, controls the trade-off between bias and variance in advantage estimation.
- **cliprange**: Typically around 0.2, limits the policy update size.
- **cliprange_value**: Similar to cliprange but for value function updates.
- **initial_kl_coef**: Initial coefficient for KL penalty.
- **target_kl**: Target KL divergence between reference and current policy.
- **kl_horizon**: Number of steps over which to adjust the KL coefficient.

### 1.7 Hyperparameter Mapping in Codebase

The following table maps the theoretical hyperparameters to their implementation in the VERL codebase:

| Theoretical Parameter | Codebase Parameter | Typical Value | Description |
|-----------------------|--------------------|---------------|-------------|
| gamma                 | `config.critic.gae.gamma` | 0.99 | Discount factor for future rewards |
| lambda                | `config.critic.gae.lam` | 0.95 | GAE parameter controlling bias-variance tradeoff |
| cliprange             | `config.actor.cliprange` | 0.2 | Clipping parameter for policy loss |
| cliprange_value       | `config.critic.cliprange_value` | 0.2 | Clipping parameter for value loss |
| initial_kl_coef       | `config.critic.kl_ctrl.kl_coef` | 0.1-0.2 | Initial KL penalty coefficient |
| target_kl             | `config.critic.kl_ctrl.target_kl` | 0.01-0.05 | Target KL divergence from reference policy |
| kl_horizon            | `config.critic.kl_ctrl.horizon` | 10000 | Steps over which to adjust KL coefficient |
| learning_rate (actor) | `config.actor.learning_rate` | 1e-5 to 5e-6 | Learning rate for actor model updates |
| learning_rate (critic)| `config.critic.learning_rate` | 1e-5 to 5e-6 | Learning rate for critic model updates |
| entropy_coef          | `config.actor.ent_coef` | 0.0 to 0.01 | Coefficient for entropy regularization |
| vf_coef               | `config.critic.vf_coef` | 0.5 to 1.0 | Coefficient for value function loss |

These parameters can be found in the configuration files and are passed to the appropriate training functions during initialization.

### 1.8 PPO Algorithm Box

Below is a comprehensive algorithm box for PPO implementation in the VERL framework:

```
Algorithm: Proximal Policy Optimization (PPO) for Language Models

INITIALIZE:
    θ_actor ← parameters of actor (policy) model
    θ_critic ← parameters of critic (value) model
    θ_ref ← copy of θ_actor (frozen reference model)
    β ← initial_kl_coef (KL penalty coefficient)
    
HYPERPARAMETERS:
    γ ← discount factor (gamma)
    λ ← GAE parameter (lambda)
    ε_clip ← clipping parameter for policy loss (cliprange)
    ε_value ← clipping parameter for value function (cliprange_value)
    c_1 ← value function coefficient (vf_coef)
    c_2 ← entropy coefficient (ent_coef)
    c_3 ← KL penalty coefficient (controlled by KL controller)
    
FOR each iteration:
    // Data Collection Phase
    Generate responses from current policy π_θ_actor
    Compute rewards for each response
    
    // Advantage Estimation
    Compute value estimates V(s_t) using critic model
    Compute advantages using GAE:
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
    Compute returns: R_t = Â_t + V(s_t)
    
    // Policy Update Phase
    FOR mini-batch in dataset:
        // Actor Update
        Compute log probabilities log_π_θ(a_t|s_t) and log_π_θ_old(a_t|s_t)
        Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        Compute policy loss:
            L^CLIP(θ) = min(r_t(θ)Â_t, clip(r_t(θ), 1-ε_clip, 1+ε_clip)Â_t)
        
        // Critic Update
        Compute value predictions V_θ(s_t) from critic model
        Compute value loss:
            L^VF(θ) = max((V_θ(s_t) - R_t)^2, (V_θ_clip(s_t) - R_t)^2)
        
        // Entropy and KL Penalty
        Compute entropy: S[π_θ](s_t)
        Compute KL divergence: KL[π_θ_ref || π_θ]
        
        // Total Loss
        L^TOTAL(θ) = -L^CLIP(θ) + c_1*L^VF(θ) - c_2*S[π_θ](s) + c_3*KL[π_θ_ref || π_θ]
        
        // Gradient Update
        Update actor parameters: θ_actor ← θ_actor - α_actor∇_θL^TOTAL(θ)
        Update critic parameters: θ_critic ← θ_critic - α_critic∇_θL^VF(θ)
    
    // KL Coefficient Update
    Measure current KL: KL_current = KL[π_θ_ref || π_θ]
    Update β to maintain target_kl:
        If KL_current > target_kl: Increase β
        If KL_current < target_kl: Decrease β
```

#### Objective Functions for Each Model Component

1. **Actor (Policy) Model**:
   - **Primary Objective**: Maximize expected rewards while staying close to the old policy
   - **Loss Function**: 
     $$L^{Actor}(\theta) = -L^{CLIP}(\theta) - c_2 S[\pi_\theta](s) + c_3 \text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

2. **Critic (Value) Model**:
   - **Primary Objective**: Accurately predict expected returns
   - **Loss Function**: 
     $$L^{Critic}(\phi) = c_1 \cdot \frac{1}{2} \max\left((V_\phi(s) - R)^2, (V_{\phi_{clip}}(s) - R)^2\right)$$

3. **Reference Model**:
   - **Role**: Provide a stable reference point to prevent policy from drifting too far
   - **Used in**: KL divergence calculation
     $$\text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

4. **Rollout Model**:
   - **Role**: Generate responses during data collection
   - **Not directly optimized**, but identical to actor model at the start of each iteration

The balance between these objectives is critical for stable and effective PPO training. Too much emphasis on the KL penalty can limit learning, while too little can lead to unstable updates. Similarly, proper weighting of the value function loss helps ensure accurate advantage estimation, which is crucial for policy improvement.

## 2. Leave-One-Out Reinforcement Learning (RLOO)

### 2.1 Overview

RLOO (Leave-One-Out Reinforcement Learning) is a method that addresses the challenge of estimating the advantage function in reinforcement learning for language models. It's particularly useful when there's no explicit value function and when each prompt has multiple responses.

### 2.2 Objective Functions

RLOO uses a simple but effective approach to calculate advantages. The key idea is to leave one response out when calculating the baseline, which prevents the model from being penalized for responses that are all good.

**Mathematical Formulation:**

For a prompt $p$ with $n$ responses, the RLOO advantage for response $i$ is calculated as:

$$A_i = r_i \cdot \frac{n}{n-1} - \mu_{-i} \cdot \frac{n}{n-1}$$

where $r_i$ is the reward for response $i$, and $\mu_{-i}$ is the mean reward of all responses except $i$ for the same prompt:

$$\mu_{-i} = \frac{1}{n-1} \sum_{j \neq i} r_j$$

The objective function is similar to PPO but with this advantage estimation:

```python
# From verl/trainer/ppo/core_algos.py
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                               eos_mask: torch.Tensor,
                               index: torch.Tensor,
                               epsilon: float = 1e-6):
    """
    Compute advantages using Leave-One-Out (RLOO) method.
    
    Parameters:
        token_level_rewards: Rewards for each token in each sequence. Shape: [batch_size, sequence_length]
        eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
        index: Tensor of prompt indices that map each response to its prompt. Shape: [batch_size]
               (Used to group responses from the same prompt together)
        epsilon: Small constant to prevent numerical instability. Shape: [] (scalar)
        
    Returns:
        scores: Calculated advantages. Shape: [batch_size, sequence_length]
        scores: Returns (identical to advantages in RLOO). Shape: [batch_size, sequence_length]
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                    1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

The overall RLOO objective function is:

$$L^{RLOO}(\theta) = \hat{E}_t \left[ \min\left(r_t(\theta) A_t^{RLOO}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t^{RLOO} \right) \right] - c_1 L^{KL}(\theta)$$

where $A_t^{RLOO}$ is the RLOO advantage estimate, and $c_1$ is the coefficient for the KL penalty.

### 2.3 Implementation Details

In RLOO, for each prompt, multiple responses are generated. When calculating the advantage for a response, the baseline is the average reward of all other responses for the same prompt. This approach provides a more accurate baseline that's specific to each prompt.

The key steps in the implementation are:
1. Group rewards by prompt index
2. Calculate the mean reward for each prompt
3. For each response, calculate the advantage as the difference between its reward and the leave-one-out mean of all other responses for the same prompt

The formula used is:
```
advantage_i = r_i * n/(n-1) - mean_rewards * n/(n-1)
```
where n is the number of responses for a prompt and mean_rewards is the mean of all responses for that prompt.

### 2.4 Key Hyperparameters

RLOO uses similar hyperparameters to PPO, but doesn't require gamma or lambda since it doesn't use GAE:

- **cliprange**: Typically around 0.2, limits the policy update size.
- **cliprange_value**: For value function updates if a critic is used.
- **initial_kl_coef**: Initial coefficient for KL penalty.
- **target_kl**: Target KL divergence between reference and current policy.

### 2.5 RLOO Algorithm Box

Below is the algorithm box for RLOO implementation in the VERL framework:

```
Algorithm: Leave-One-Out Reinforcement Learning (RLOO) for Language Models

INITIALIZE:
    θ_actor ← parameters of actor (policy) model
    θ_ref ← copy of θ_actor (frozen reference model)
    β ← initial_kl_coef (KL penalty coefficient)
    
HYPERPARAMETERS:
    ε_clip ← clipping parameter for policy loss (cliprange)
    c_KL ← KL penalty coefficient (controlled by KL controller)
    
FOR each iteration:
    // Data Collection Phase
    Generate multiple responses for each prompt from current policy π_θ_actor
    Compute rewards for each response
    
    // Advantage Estimation Phase
    Group responses by prompt
    FOR each prompt p:
        FOR each response i of prompt p:
            Compute mean reward of all other responses: μ_{-i} = (1/(n-1)) ∑_{j≠i} r_j
            Compute RLOO advantage: A_i = r_i * n/(n-1) - μ_{-i} * n/(n-1)
    
    // Policy Update Phase
    FOR mini-batch in dataset:
        // Actor Update
        Compute log probabilities log_π_θ(a_t|s_t) and log_π_θ_old(a_t|s_t)
        Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        Compute policy loss:
            L^CLIP(θ) = min(r_t(θ)A_t^RLOO, clip(r_t(θ), 1-ε_clip, 1+ε_clip)A_t^RLOO)
        
        // KL Penalty
        Compute KL divergence: KL[π_θ_ref || π_θ]
        
        // Total Loss
        L^TOTAL(θ) = -L^CLIP(θ) + c_KL*KL[π_θ_ref || π_θ]
        
        // Gradient Update
        Update actor parameters: θ_actor ← θ_actor - α_actor∇_θL^TOTAL(θ)
    
    // KL Coefficient Update
    Measure current KL: KL_current = KL[π_θ_ref || π_θ]
    Update β to maintain target_kl:
        If KL_current > target_kl: Increase β
        If KL_current < target_kl: Decrease β
```

#### Objective Functions for RLOO

1. **Actor (Policy) Model**:
   - **Primary Objective**: Maximize expected rewards while staying close to the old policy
   - **Loss Function**: 
     $$L^{Actor}(\theta) = -L^{CLIP}(\theta) + c_{KL} \text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

2. **Reference Model**:
   - **Role**: Provide a stable reference point to prevent policy from drifting too far
   - **Used in**: KL divergence calculation
     $$\text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

The key advantage of RLOO is its robust advantage estimation when multiple responses are generated for each prompt, without requiring a separate value function.

## 3. Generalized Reward Policy Optimization (GRPO)

### 3.1 Overview

GRPO (Generalized Reward Policy Optimization) is similar to RLOO but takes a slightly different approach to advantage estimation. It uses standardization (z-scoring) of rewards within each prompt to compute advantages.

### 3.2 Objective Functions

**Mathematical Formulation:**

For a prompt $p$ with multiple responses, the GRPO advantage for response $i$ is the z-score of its reward:

$$A_i = \frac{r_i - \mu_p}{\sigma_p + \epsilon}$$

where $r_i$ is the reward for response $i$, $\mu_p$ is the mean reward for prompt $p$, $\sigma_p$ is the standard deviation of rewards for prompt $p$, and $\epsilon$ is a small constant to prevent division by zero.

GRPO uses a standardized advantage function:

```python
# From verl/trainer/ppo/core_algos.py
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                               eos_mask: torch.Tensor,
                               index: torch.Tensor,
                               epsilon: float = 1e-6):
    """
    Compute advantages using Generalized Reward Policy Optimization (GRPO) method.
    
    Parameters:
        token_level_rewards: Rewards for each token in each sequence. Shape: [batch_size, sequence_length]
        eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
        index: Tensor of prompt indices that map each response to its prompt. Shape: [batch_size]
               (Used to group responses from the same prompt together)
        epsilon: Small constant to prevent division by zero in standardization. Shape: [] (scalar)
        
    Returns:
        scores: Calculated advantages using z-score standardization. Shape: [batch_size, sequence_length]
        scores: Returns (identical to advantages in GRPO). Shape: [batch_size, sequence_length]
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

The overall GRPO objective function is:

$$L^{GRPO}(\theta) = \hat{E}_t \left[ \min\left(r_t(\theta) A_t^{GRPO}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t^{GRPO} \right) \right] - c_1 L^{KL}(\theta)$$

where $A_t^{GRPO}$ is the GRPO advantage estimate, and $c_1$ is the coefficient for the KL penalty.

### 3.3 Implementation Details

GRPO standardizes rewards by calculating the mean and standard deviation of rewards for each prompt, then computing the z-score for each response. This helps normalize the advantages across different prompts.

The key steps are:
1. Group rewards by prompt index
2. Calculate mean and standard deviation of rewards for each prompt
3. Standardize each reward: (reward - mean) / (std + epsilon)

The standardization helps handle varying reward scales across different prompts and ensures that the policy update is not dominated by prompts with larger reward magnitudes.

### 3.4 Key Hyperparameters

GRPO uses similar hyperparameters to PPO:

- **cliprange**: Typically around 0.2, limits the policy update size.
- **initial_kl_coef**: Initial coefficient for KL penalty (beta).
- **target_kl**: Target KL divergence between reference and current policy.
- **epsilon**: Small constant (e.g., 1e-6) added to standard deviation to prevent division by zero.

### 3.5 GRPO Algorithm Box

Below is the algorithm box for GRPO implementation in the VERL framework:

```
Algorithm: Generalized Reward Policy Optimization (GRPO) for Language Models

INITIALIZE:
    θ_actor ← parameters of actor (policy) model
    θ_ref ← copy of θ_actor (frozen reference model)
    β ← initial_kl_coef (KL penalty coefficient)
    
HYPERPARAMETERS:
    ε_clip ← clipping parameter for policy loss (cliprange)
    c_KL ← KL penalty coefficient (controlled by KL controller)
    ε ← small constant to prevent division by zero
    
FOR each iteration:
    // Data Collection Phase
    Generate multiple responses for each prompt from current policy π_θ_actor
    Compute rewards for each response
    
    // Advantage Estimation Phase
    Group responses by prompt
    FOR each prompt p:
        Compute mean reward: μ_p = (1/n) ∑_i r_i
        Compute standard deviation: σ_p = sqrt((1/n) ∑_i (r_i - μ_p)²)
        FOR each response i of prompt p:
            Compute GRPO advantage: A_i = (r_i - μ_p)/(σ_p + ε)
    
    // Policy Update Phase
    FOR mini-batch in dataset:
        // Actor Update
        Compute log probabilities log_π_θ(a_t|s_t) and log_π_θ_old(a_t|s_t)
        Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        Compute policy loss:
            L^CLIP(θ) = min(r_t(θ)A_t^GRPO, clip(r_t(θ), 1-ε_clip, 1+ε_clip)A_t^GRPO)
        
        // KL Penalty
        Compute KL divergence: KL[π_θ_ref || π_θ]
        
        // Total Loss
        L^TOTAL(θ) = -L^CLIP(θ) + c_KL*KL[π_θ_ref || π_θ]
        
        // Gradient Update
        Update actor parameters: θ_actor ← θ_actor - α_actor∇_θL^TOTAL(θ)
    
    // KL Coefficient Update
    Measure current KL: KL_current = KL[π_θ_ref || π_θ]
    Update β to maintain target_kl:
        If KL_current > target_kl: Increase β
        If KL_current < target_kl: Decrease β
```

#### Objective Functions for GRPO

1. **Actor (Policy) Model**:
   - **Primary Objective**: Maximize expected rewards while staying close to the old policy
   - **Loss Function**: 
     $$L^{Actor}(\theta) = -L^{CLIP}(\theta) + c_{KL} \text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

2. **Reference Model**:
   - **Role**: Provide a stable reference point to prevent policy from drifting too far
   - **Used in**: KL divergence calculation
     $$\text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

The key feature of GRPO is the standardization of rewards across prompts, which helps handle varying reward scales and ensures that policy updates are more balanced.

## 4. REINFORCE++

### 4.1 Overview

REINFORCE++ is an extension of the classic REINFORCE algorithm with improvements for language model training. It uses a token-level credit assignment mechanism to distribute rewards more effectively throughout the sequence.

### 4.2 Objective Functions

**Mathematical Formulation:**

REINFORCE++ computes the return at each time step $t$ as the discounted sum of future rewards:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

where $\gamma$ is the discount factor, $r_k$ is the reward at time $k$, and $T$ is the final time step.

The advantages are then the standardized returns:

$$A_t = \frac{G_t - \mu(G)}{\sigma(G) + \epsilon}$$

where $\mu(G)$ and $\sigma(G)$ are the mean and standard deviation of the returns.

REINFORCE++ calculates advantages by propagating rewards backward through time:

```python
# From verl/trainer/ppo/core_algos.py
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                              gamma: torch.Tensor):
    """
    Compute advantages using REINFORCE++ method by propagating rewards backward through time.
    
    Parameters:
        token_level_rewards: Rewards for each token in each sequence. Shape: [batch_size, sequence_length]
        eos_mask: Binary mask indicating which tokens are valid (1) vs padding (0). Shape: [batch_size, sequence_length]
        gamma: Discount factor for future rewards. Shape: [] (scalar)
        
    Returns:
        advantages: Calculated advantages (standardized returns). Shape: [batch_size, sequence_length]
        returns: Calculated returns before standardization. Shape: [batch_size, sequence_length]
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns
```

The overall REINFORCE++ objective function is:

$$L^{REINFORCE++}(\theta) = -\hat{E}_t \left[ \log \pi_\theta(a_t|s_t) A_t^{REINFORCE++} \right] - c_1 L^{KL}(\theta)$$

where $A_t^{REINFORCE++}$ is the REINFORCE++ advantage estimate, and $c_1$ is the coefficient for the KL penalty.

### 4.3 Implementation Details

REINFORCE++ has these key features:

1. **Backward Credit Assignment**: Propagates rewards backward through the sequence, assigning credit to tokens based on their contribution to future rewards.

2. **Standardization**: The advantages are standardized (whitened) to have zero mean and unit variance, which helps with training stability.

3. **Discount Factor**: Uses a discount factor (gamma) to control the influence of future rewards on earlier tokens.

The core idea is that each token's advantage is based on the discounted sum of future rewards, which helps address the temporal credit assignment problem in sequence generation.

### 4.4 Key Hyperparameters

- **gamma**: Discount factor (typically around 0.99) that controls how much future rewards affect earlier tokens.
- **cliprange**: Typically around 0.2, limits the policy update size.
- **initial_kl_coef**: Initial coefficient for KL penalty.
- **target_kl**: Target KL divergence between reference and current policy.

### 4.5 REINFORCE++ Algorithm Box

Below is the algorithm box for REINFORCE++ implementation in the VERL framework:

```
Algorithm: REINFORCE++ for Language Models

INITIALIZE:
    θ_actor ← parameters of actor (policy) model
    θ_ref ← copy of θ_actor (frozen reference model)
    
HYPERPARAMETERS:
    γ ← discount factor
    ε_clip ← clipping parameter for policy loss
    c_KL ← KL penalty coefficient
    
FOR each iteration:
    // Data Collection Phase
    Generate responses from current policy π_θ_actor
    Compute rewards for each response
    
    // Advantage Estimation
    Compute returns G_t for each time step t
    Compute standardized advantages A_t = (G_t - μ(G)) / (σ(G) + ε)
    
    // Policy Update Phase
    FOR mini-batch in dataset:
        // Compute log probabilities
        Calculate log π_θ(a_t|s_t) for each action a_t in the batch
        Calculate log π_ref(a_t|s_t) for each action a_t using reference model
        
        // Compute log ratios
        Calculate r_t(θ) = log π_θ(a_t|s_t) - log π_ref(a_t|s_t)
        
        // Compute REINFORCE++ loss
        L^REINFORCE++ = -Σ_t (log π_θ(a_t|s_t) * A_t)
        
        // Gradient Update
        Update actor parameters: θ_actor ← θ_actor - α * ∇_θ L^REINFORCE++
```

#### Objective Functions for REINFORCE++

1. **Actor (Policy) Model**:
   - **Primary Objective**: Maximize expected rewards while staying close to the old policy
   - **Loss Function**: 
     $$L^{REINFORCE++}(\theta) = -\hat{E}_t \left[ \log \pi_\theta(a_t|s_t) A_t^{REINFORCE++} \right]$$

2. **Reference Model**:
   - **Role**: Provide a stable reference point to prevent policy from drifting too far
   - **Used in**: KL divergence calculation
     $$\text{KL}[\pi_{\theta_{ref}} || \pi_\theta]$$

The key feature of REINFORCE++ is the backward credit assignment and standardization of advantages, which helps with training stability and encourages exploration.