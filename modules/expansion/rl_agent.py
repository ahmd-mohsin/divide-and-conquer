# rl_agent.py
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from expansion_config import ExpansionConfig
from rl_environment import CoTExpansionEnv
from expansion_schemas import SubProblemExpansion, ExpansionResult, CoTChain
from division.schemas import SubProblem, Decomposition


class HierarchicalRLAgent:
    """
    Hierarchical RL agent using PPO.
    
    Learns to select diverse reasoning strategies for each sub-problem.
    High entropy coefficient encourages exploration of different strategies.
    """
    
    def __init__(self, config: ExpansionConfig):
        self.config = config
        self.model: Optional[PPO] = None
        self.env: Optional[DummyVecEnv] = None
        
        # Create directories
        Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(config.results_path).mkdir(parents=True, exist_ok=True)
    
    def create_environment(self, subproblems: List[SubProblem], n_envs: int = 1) -> DummyVecEnv:
        """
        Create vectorized environment.
        
        Args:
            subproblems: List of sub-problems to practice on
            n_envs: Number of parallel environments
            
        Returns:
            Vectorized environment
        """
        if n_envs == 1:
            # Single environment
            env = CoTExpansionEnv(self.config, subproblems)
            env = Monitor(env)
            vec_env = DummyVecEnv([lambda: env])
        else:
            # Multiple parallel environments
            def make_env():
                env = CoTExpansionEnv(self.config, subproblems)
                return Monitor(env)
            
            vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        return vec_env
    
    def create_model(self, env: DummyVecEnv) -> PPO:
        """
        Create PPO model with configured hyperparameters.
        
        Key settings for exploration:
        - High ent_coef (entropy coefficient) encourages trying different strategies
        - Temperature in action sampling adds randomness
        """
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config.rl.learning_rate,
            n_steps=self.config.rl.n_steps,
            batch_size=self.config.rl.batch_size,
            n_epochs=self.config.rl.n_epochs,
            gamma=self.config.rl.gamma,
            gae_lambda=self.config.rl.gae_lambda,
            clip_range=self.config.rl.clip_range,
            ent_coef=self.config.rl.ent_coef,  # HIGH = more exploration
            vf_coef=self.config.rl.vf_coef,
            max_grad_norm=self.config.rl.max_grad_norm,
            verbose=1 if self.config.verbose else 0,
            tensorboard_log=f"{self.config.results_path}/tensorboard/"
        )
        
        return model
    
    def train(
        self,
        subproblems: List[SubProblem],
        total_timesteps: Optional[int] = None,
        n_envs: int = 1
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            subproblems: Sub-problems to train on
            total_timesteps: Total training steps (uses config default if None)
            n_envs: Number of parallel environments
            
        Returns:
            Training statistics
        """
        timesteps = total_timesteps or self.config.rl.total_timesteps
        
        print(f"\n{'='*60}")
        print(f"Training PPO Agent for CoT Expansion")
        print(f"{'='*60}")
        print(f"Sub-problems: {len(subproblems)}")
        print(f"Strategies: {len(self.config.rl.exploration_strategies)}")
        print(f"Total timesteps: {timesteps}")
        print(f"Entropy coefficient: {self.config.rl.ent_coef} (higher = more exploration)")
        print(f"{'='*60}\n")
        
        # Create environment
        self.env = self.create_environment(subproblems, n_envs=n_envs)
        
        # Create model
        self.model = self.create_model(self.env)
        
        # Callbacks for saving checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.rl.save_freq,
            save_path=self.config.model_save_path,
            name_prefix="ppo_cot_agent"
        )
        
        # Train
        self.model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{self.config.model_save_path}/ppo_cot_agent_final"
        self.model.save(final_path)
        print(f"\n✓ Model saved to {final_path}")
        
        return {
            "total_timesteps": timesteps,
            "final_model_path": final_path,
        }
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = PPO.load(path)
        print(f"✓ Loaded model from {path}")
    
    def expand_subproblem(
        self,
        subproblem: SubProblem,
        use_learned_policy: bool = True
    ) -> SubProblemExpansion:
        """
        Generate multiple diverse chains for a single sub-problem.
        
        Args:
            subproblem: The sub-problem to expand
            use_learned_policy: If True, use trained RL agent. If False, use random strategies.
            
        Returns:
            SubProblemExpansion with all generated chains
        """
        # Create temporary environment
        env = CoTExpansionEnv(self.config, [subproblem])
        obs, _ = env.reset()
        
        chains = []
        
        for i in range(self.config.cot.num_chains):
            if use_learned_policy and self.model is not None:
                # Use trained agent to select strategy
                action, _ = self.model.predict(obs, deterministic=False)
            else:
                # Random strategy selection
                action = env.action_space.sample()
            
            # Generate chain
            obs, reward, done, truncated, info = env.step(action)
            
            if self.config.verbose:
                print(f"Chain {i+1}/{self.config.cot.num_chains}: "
                      f"strategy={info['strategy']}, reward={reward:.3f}")
        
        # Get all generated chains
        chains = env.get_episode_chains()
        
        # Compute diversity matrix
        from diversity_metrics import DiversityMetrics
        diversity_metrics = DiversityMetrics()
        chain_texts = [c.get_text() for c in chains]
        diversity_matrix = diversity_metrics.compute_pairwise_diversity(chain_texts)
        
        # Create expansion result
        expansion = SubProblemExpansion(
            subproblem_id=subproblem.id,
            subproblem_goal=subproblem.goal,
            chains=chains,
            diversity_matrix=diversity_matrix.tolist()
        )
        
        # Identify best chain
        if chains:
            best_chain = max(chains, key=lambda c: c.rl_reward or 0.0)
            expansion.best_chain_id = best_chain.chain_id
        
        return expansion
    
    def expand_decomposition(
        self,
        decomposition: Decomposition,
        use_learned_policy: bool = True
    ) -> ExpansionResult:
        """
        Expand all sub-problems in a decomposition.
        
        Args:
            decomposition: The problem decomposition from Module 1
            use_learned_policy: Whether to use trained RL agent
            
        Returns:
            Complete expansion result
        """
        print(f"\n{'='*60}")
        print(f"Expanding Decomposition")
        print(f"{'='*60}")
        print(f"Problem: {decomposition.problem[:80]}")
        print(f"Sub-problems: {len(decomposition.nodes)}")
        print(f"Chains per sub-problem: {self.config.cot.num_chains}")
        print(f"Policy: {'Learned RL' if use_learned_policy else 'Random'}")
        print(f"{'='*60}\n")
        
        subproblem_expansions = []
        
        for i, node in enumerate(decomposition.nodes, 1):
            print(f"\n--- Sub-problem {i}/{len(decomposition.nodes)} ---")
            print(f"Goal: {node.goal[:60]}")
            
            expansion = self.expand_subproblem(node, use_learned_policy)
            subproblem_expansions.append(expansion)
            
            # Print stats
            avg_reward = np.mean([c.rl_reward for c in expansion.chains if c.rl_reward])
            avg_diversity = np.mean(expansion.diversity_matrix) if expansion.diversity_matrix else 0.0
            print(f"✓ Generated {len(expansion.chains)} chains")
            print(f"  Avg reward: {avg_reward:.3f}")
            print(f"  Avg diversity: {avg_diversity:.3f}")
        
        # Create result
        result = ExpansionResult(
            decomposition_id=f"decomp_{id(decomposition)}",
            problem=decomposition.problem,
            subproblem_expansions=subproblem_expansions,
            total_chains=sum(len(exp.chains) for exp in subproblem_expansions),
            avg_diversity=np.mean([
                np.mean(exp.diversity_matrix) if exp.diversity_matrix else 0.0
                for exp in subproblem_expansions
            ]),
            avg_quality=np.mean([
                np.mean([c.rl_reward for c in exp.chains if c.rl_reward])
                for exp in subproblem_expansions
            ])
        )
        
        print(f"\n{'='*60}")
        print(f"Expansion Complete")
        print(f"{'='*60}")
        print(f"Total chains: {result.total_chains}")
        print(f"Avg diversity: {result.avg_diversity:.3f}")
        print(f"Avg quality: {result.avg_quality:.3f}")
        print(f"{'='*60}\n")
        
        return result