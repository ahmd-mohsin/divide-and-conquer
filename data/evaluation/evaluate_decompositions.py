#!/usr/bin/env python3
"""
Decomposition Quality Evaluation (Independent Sampling)
Evaluates 100 problems from each model independently using three metric categories.

Place in: ~/Divide-and-Conquer/data/evaluation/
Run from: ~/Divide-and-Conquer/data/
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass, asdict
import networkx as nx
from difflib import SequenceMatcher


@dataclass
class DAGMetrics:
    """Category 1: DAG Structural Quality"""
    has_cycle: bool
    topo_sort_exists: bool
    depth: int
    breadth: float
    work: int
    span: int
    parallelism: float
    critical_path_token_share: float
    justified_edge_rate: float
    avg_step_size_tokens: float
    step_size_cv: float
    duplicate_goal_rate: float


@dataclass
class FaithfulnessMetrics:
    """Category 2: Faithfulness & Minimality"""
    redundancy_rate: float
    avg_dependency_usage: float
    specificity_score: float


@dataclass
class LFCMetrics:
    """Category 3: Compositional Skill"""
    type_consistency: float
    closure_rate: float
    skill_diversity: float
    reuse_potential: float
    lfc_score: float


@dataclass
class ProblemEvaluation:
    """Evaluation for one problem"""
    problem_id: str
    problem_text: str
    num_subproblems: int
    dag_metrics: DAGMetrics
    faithfulness_metrics: FaithfulnessMetrics
    lfc_metrics: LFCMetrics
    overall_score: float


class DecompositionAnalyzer:
    """Analyze decomposition quality"""
    
    def __init__(self):
        self.vague_words = {
            'analyze', 'consider', 'discuss', 'examine', 'explore',
            'investigate', 'look', 'review', 'study', 'think'
        }
        self.concrete_words = {
            'calculate', 'compute', 'subtract', 'add', 'multiply',
            'divide', 'solve', 'find', 'determine', 'identify'
        }
    
    def build_dag(self, problem_data: Dict) -> nx.DiGraph:
        """Build DAG from problem decomposition"""
        G = nx.DiGraph()
        if not problem_data.get('chains'):
            return G
        
        steps = problem_data['chains'][0]['steps']
        for step in steps:
            node_id = step['subproblem_id']
            G.add_node(node_id, **step)
            for dep_id in step.get('dependencies_used', []):
                G.add_edge(dep_id, node_id)
        
        return G
    
    def compute_dag_metrics(self, G: nx.DiGraph, problem_data: Dict) -> DAGMetrics:
        """Compute DAG Structural Quality metrics"""
        if len(G) == 0:
            return DAGMetrics(False, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        has_cycle = not nx.is_directed_acyclic_graph(G)
        topo_sort_exists = not has_cycle
        work = G.number_of_nodes()
        
        try:
            if has_cycle:
                depth = span = 0
            else:
                longest_path = nx.dag_longest_path_length(G)
                depth = span = longest_path + 1
        except:
            depth = span = 1
        
        parallelism = work / max(span, 1)
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        breadth = np.mean(out_degrees) if out_degrees else 0
        
        step_sizes = [len(G.nodes[n].get('reasoning', '').split()) for n in G.nodes()]
        avg_step_size = np.mean(step_sizes) if step_sizes else 0
        step_size_cv = np.std(step_sizes) / max(avg_step_size, 1) if avg_step_size > 0 else 0
        
        if not has_cycle and len(G) > 0:
            try:
                longest_path_nodes = nx.dag_longest_path(G)
                critical_tokens = sum(len(G.nodes[n].get('reasoning', '').split()) for n in longest_path_nodes)
                total_tokens = sum(step_sizes)
                critical_path_token_share = critical_tokens / max(total_tokens, 1)
            except:
                critical_path_token_share = 0
        else:
            critical_path_token_share = 0
        
        justified_edges = 0
        for u, v in G.edges():
            v_reasoning = G.nodes[v].get('reasoning', '').lower()
            u_answer = G.nodes[u].get('answer', '').lower()
            if u_answer and (u_answer[:20] in v_reasoning or f"step {u}" in v_reasoning):
                justified_edges += 1
        justified_edge_rate = justified_edges / max(G.number_of_edges(), 1)
        
        goals = [G.nodes[n].get('goal', '').lower() for n in G.nodes()]
        duplicate_pairs = sum(
            1 for i in range(len(goals)) for j in range(i+1, len(goals))
            if SequenceMatcher(None, goals[i], goals[j]).ratio() > 0.8
        )
        total_pairs = len(goals) * (len(goals) - 1) // 2
        duplicate_goal_rate = duplicate_pairs / max(total_pairs, 1)
        
        return DAGMetrics(
            has_cycle, topo_sort_exists, depth, breadth, work, span, parallelism,
            critical_path_token_share, justified_edge_rate, avg_step_size,
            step_size_cv, duplicate_goal_rate
        )
    
    def compute_faithfulness_metrics(self, G: nx.DiGraph, problem_data: Dict) -> FaithfulnessMetrics:
        """Compute Faithfulness & Minimality metrics"""
        if len(G) == 0:
            return FaithfulnessMetrics(0, 0, 0)
        
        chains = problem_data.get('chains', [])
        step_answer_variance = []
        
        if len(chains) > 1:
            answers_by_step = defaultdict(list)
            for chain in chains:
                for step in chain['steps']:
                    answers_by_step[step['subproblem_id']].append(step.get('answer', ''))
            
            for step_id, answers in answers_by_step.items():
                unique_answers = len(set(answers))
                redundancy = 1.0 - (unique_answers / max(len(answers), 1))
                step_answer_variance.append(redundancy)
        
        redundancy_rate = np.mean(step_answer_variance) if step_answer_variance else 0
        
        dep_usage_scores = []
        for node in G.nodes():
            reasoning = G.nodes[node].get('reasoning', '').lower()
            deps_used = G.nodes[node].get('dependencies_used', [])
            
            if deps_used:
                mentions = sum(
                    1 for dep_id in deps_used
                    if dep_id in G.nodes and (
                        G.nodes[dep_id].get('answer', '').lower()[:20] in reasoning or
                        f"step {dep_id}" in reasoning
                    )
                )
                dep_usage_scores.append(mentions / len(deps_used))
        
        avg_dependency_usage = np.mean(dep_usage_scores) if dep_usage_scores else 0
        
        specificity_scores = []
        for node in G.nodes():
            text = (G.nodes[node].get('goal', '') + " " + G.nodes[node].get('plan', '')).lower()
            words = set(text.split())
            if len(words) > 0:
                vague_count = len(words & self.vague_words)
                concrete_count = len(words & self.concrete_words)
                specificity = max(0, min(1, (concrete_count - vague_count) / len(words) + 0.5))
            else:
                specificity = 0.5
            specificity_scores.append(specificity)
        
        specificity_score = np.mean(specificity_scores) if specificity_scores else 0.5
        
        return FaithfulnessMetrics(redundancy_rate, avg_dependency_usage, specificity_score)
    
    def compute_lfc_metrics(self, G: nx.DiGraph, problem_data: Dict) -> LFCMetrics:
        """Compute Compositional Skill metrics"""
        if len(G) == 0:
            return LFCMetrics(0, 0, 0, 0, 0)
        
        skill_types = {
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', 'sum', 'total', 'calculate'],
            'extraction': ['identify', 'extract', 'find', 'given', 'initial'],
            'comparison': ['compare', 'greater', 'less', 'equal', 'difference'],
            'application': ['apply', 'use', 'substitute', 'formula'],
            'verification': ['check', 'verify', 'validate', 'confirm']
        }
        
        step_types = {}
        for node in G.nodes():
            text = (G.nodes[node].get('goal', '') + " " + G.nodes[node].get('plan', '')).lower()
            matched_type = 'unknown'
            for skill_type, keywords in skill_types.items():
                if any(kw in text for kw in keywords):
                    matched_type = skill_type
                    break
            step_types[node] = matched_type
        
        compatible_pairs = {
            ('extraction', 'arithmetic'), ('extraction', 'comparison'),
            ('arithmetic', 'arithmetic'), ('arithmetic', 'comparison'),
            ('arithmetic', 'application'), ('comparison', 'verification'),
            ('application', 'verification')
        }
        
        consistent_edges = sum(
            1 for u, v in G.edges()
            if (step_types.get(u, 'unknown'), step_types.get(v, 'unknown')) in compatible_pairs
            or step_types.get(u) == step_types.get(v)
        )
        type_consistency = consistent_edges / max(G.number_of_edges(), 1)
        
        consumed_nodes = set(u for u, v in G.edges())
        closure_rate = len(consumed_nodes) / max(len(G), 1)
        
        type_counts = Counter(step_types.values())
        total_steps = len(step_types)
        if total_steps > 0:
            entropy = -sum((c/total_steps) * np.log2(c/total_steps) for c in type_counts.values() if c > 0)
            max_entropy = np.log2(len(skill_types))
            skill_diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            skill_diversity = 0
        
        import re
        goal_templates = []
        for node in G.nodes():
            goal = G.nodes[node].get('goal', '').lower()
            template = re.sub(r'\d+', 'N', goal)
            template = re.sub(r'\$[\d.]+', '$N', template)
            goal_templates.append(template)
        
        unique_templates = len(set(goal_templates))
        reuse_potential = 1.0 - (unique_templates / max(len(goal_templates), 1))
        
        lfc_score = 0.25 * (type_consistency + closure_rate + skill_diversity + reuse_potential)
        
        return LFCMetrics(type_consistency, closure_rate, skill_diversity, reuse_potential, lfc_score)
    
    def compute_overall_score(self, dag: DAGMetrics, faith: FaithfulnessMetrics, lfc: LFCMetrics) -> float:
        """Compute overall quality score (0-1, higher = better)"""
        dag_score = (
            (1.0 if dag.topo_sort_exists else 0.0) * 0.3 +
            (dag.parallelism / max(dag.work, 1)) * 0.2 +
            dag.justified_edge_rate * 0.2 +
            (1.0 - dag.duplicate_goal_rate) * 0.15 +
            (1.0 - dag.step_size_cv) * 0.15
        ) * 0.4
        
        faith_score = (
            (1.0 - faith.redundancy_rate) * 0.4 +
            faith.avg_dependency_usage * 0.3 +
            faith.specificity_score * 0.3
        ) * 0.3
        
        lfc_score = lfc.lfc_score * 0.3
        
        return dag_score + faith_score + lfc_score
    
    def evaluate_problem(self, problem_data: Dict) -> ProblemEvaluation:
        """Evaluate a single problem"""
        G = self.build_dag(problem_data)
        
        dag_metrics = self.compute_dag_metrics(G, problem_data)
        faith_metrics = self.compute_faithfulness_metrics(G, problem_data)
        lfc_metrics = self.compute_lfc_metrics(G, problem_data)
        overall_score = self.compute_overall_score(dag_metrics, faith_metrics, lfc_metrics)
        
        return ProblemEvaluation(
            problem_id=problem_data.get('decomposition_id', 'unknown'),
            problem_text=problem_data.get('problem', '')[:100] + "...",
            num_subproblems=problem_data.get('num_subproblems', 0),
            dag_metrics=dag_metrics,
            faithfulness_metrics=faith_metrics,
            lfc_metrics=lfc_metrics,
            overall_score=overall_score
        )


def load_and_sample_dataset(chains_dir: Path, num_samples: int = 100) -> List[Dict]:
    """Load dataset and sample N problems"""
    problem_files = sorted(list(chains_dir.glob("problem_*.json")))
    
    if len(problem_files) < num_samples:
        print(f"  Warning: Only {len(problem_files)} problems available, using all")
        sampled_files = problem_files
    else:
        sampled_files = random.sample(problem_files, num_samples)
    
    problems = []
    for problem_file in sampled_files:
        with open(problem_file, 'r') as f:
            problems.append(json.load(f))
    
    return problems


def aggregate_evaluations(evaluations: List[ProblemEvaluation]) -> Dict:
    """Aggregate evaluation metrics"""
    if not evaluations:
        return {}
    
    def avg_attr(attr_path: str) -> float:
        values = []
        for eval_result in evaluations:
            obj = eval_result
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, bool):
                obj = 1.0 if obj else 0.0
            values.append(float(obj))
        return np.mean(values)
    
    return {
        "total_problems": len(evaluations),
        "avg_subproblems": avg_attr("num_subproblems"),
        "dag_metrics": {
            "avg_depth": avg_attr("dag_metrics.depth"),
            "avg_parallelism": avg_attr("dag_metrics.parallelism"),
            "avg_justified_edge_rate": avg_attr("dag_metrics.justified_edge_rate"),
            "avg_duplicate_goal_rate": avg_attr("dag_metrics.duplicate_goal_rate"),
            "pct_valid_dags": avg_attr("dag_metrics.topo_sort_exists") * 100
        },
        "faithfulness_metrics": {
            "avg_redundancy_rate": avg_attr("faithfulness_metrics.redundancy_rate"),
            "avg_dependency_usage": avg_attr("faithfulness_metrics.avg_dependency_usage"),
            "avg_specificity": avg_attr("faithfulness_metrics.specificity_score")
        },
        "lfc_metrics": {
            "avg_type_consistency": avg_attr("lfc_metrics.type_consistency"),
            "avg_skill_diversity": avg_attr("lfc_metrics.skill_diversity"),
            "avg_lfc_score": avg_attr("lfc_metrics.lfc_score")
        },
        "overall_score": avg_attr("overall_score")
    }


def print_summary(model_name: str, aggregate: Dict):
    """Print evaluation summary"""
    print(f"\n{'='*80}")
    print(f"{model_name} - EVALUATION SUMMARY")
    print('='*80)
    
    print(f"\nProblems evaluated: {aggregate['total_problems']}")
    print(f"Avg subproblems: {aggregate['avg_subproblems']:.1f}")
    
    print(f"\n--- DAG Structural Quality ---")
    print(f"  Valid DAGs: {aggregate['dag_metrics']['pct_valid_dags']:.1f}%")
    print(f"  Avg depth: {aggregate['dag_metrics']['avg_depth']:.2f}")
    print(f"  Avg parallelism: {aggregate['dag_metrics']['avg_parallelism']:.2f}")
    print(f"  Edge justification rate: {aggregate['dag_metrics']['avg_justified_edge_rate']:.2%}")
    print(f"  Duplicate goal rate: {aggregate['dag_metrics']['avg_duplicate_goal_rate']:.2%}")
    
    print(f"\n--- Faithfulness & Minimality ---")
    print(f"  Redundancy rate: {aggregate['faithfulness_metrics']['avg_redundancy_rate']:.2%}")
    print(f"  Dependency usage: {aggregate['faithfulness_metrics']['avg_dependency_usage']:.2%}")
    print(f"  Specificity score: {aggregate['faithfulness_metrics']['avg_specificity']:.2%}")
    
    print(f"\n--- Compositional Skill (LFC) ---")
    print(f"  Type consistency: {aggregate['lfc_metrics']['avg_type_consistency']:.2%}")
    print(f"  Skill diversity: {aggregate['lfc_metrics']['avg_skill_diversity']:.2%}")
    print(f"  LFC score: {aggregate['lfc_metrics']['avg_lfc_score']:.3f}")
    
    print(f"\n--- Overall Quality Score ---")
    print(f"  Score (0-1): {aggregate['overall_score']:.3f}")
    print('='*80)


def main():
    # Configuration
    chains_dir1 = Path("/home/ahmed/Divide-and-Conquer/data/chains_calc_gsm8k/chains")
    chains_dir2 = Path("/home/ahmed/Divide-and-Conquer/data/gsm8k_deepseek_r1/chains")
    model1_name = "Qwen-7B"
    model2_name = "DeepSeek-R1-14B"
    num_samples = 100
    
    print("="*80)
    print("INDEPENDENT DECOMPOSITION EVALUATION")
    print("="*80)
    print(f"\nModel 1: {model1_name}")
    print(f"  Path: {chains_dir1}")
    print(f"\nModel 2: {model2_name}")
    print(f"  Path: {chains_dir2}")
    print(f"\nSamples per model: {num_samples}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    analyzer = DecompositionAnalyzer()
    
    # Evaluate Model 1
    print(f"\n{'='*80}")
    print(f"Evaluating {model1_name}...")
    print('='*80)
    
    problems1 = load_and_sample_dataset(chains_dir1, num_samples)
    print(f"Loaded {len(problems1)} problems")
    
    evaluations1 = []
    for i, problem_data in enumerate(problems1, 1):
        print(f"  [{i}/{len(problems1)}] {problem_data.get('problem', '')[:60]}...")
        evaluations1.append(analyzer.evaluate_problem(problem_data))
    
    aggregate1 = aggregate_evaluations(evaluations1)
    print_summary(model1_name, aggregate1)
    
    # Evaluate Model 2
    print(f"\n{'='*80}")
    print(f"Evaluating {model2_name}...")
    print('='*80)
    
    problems2 = load_and_sample_dataset(chains_dir2, num_samples)
    print(f"Loaded {len(problems2)} problems")
    
    evaluations2 = []
    for i, problem_data in enumerate(problems2, 1):
        print(f"  [{i}/{len(problems2)}] {problem_data.get('problem', '')[:60]}...")
        evaluations2.append(analyzer.evaluate_problem(problem_data))
    
    aggregate2 = aggregate_evaluations(evaluations2)
    print_summary(model2_name, aggregate2)
    
    # Save results
    output_file = Path("evaluation/results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    results = {
        "metadata": {
            "num_samples_per_model": num_samples,
            "random_seed": 42
        },
        model1_name: {
            "summary": aggregate1,
            "detailed": [asdict(e) for e in evaluations1]
        },
        model2_name: {
            "summary": aggregate2,
            "detailed": [asdict(e) for e in evaluations2]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_file}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print('='*80)
    
    score1 = aggregate1['overall_score']
    score2 = aggregate2['overall_score']
    
    print(f"\nOverall Scores:")
    print(f"  {model1_name}: {score1:.3f}")
    print(f"  {model2_name}: {score2:.3f}")
    
    if score1 > score2:
        print(f"\n🏆 WINNER: {model1_name} (+{score1-score2:.3f})")
    elif score2 > score1:
        print(f"\n🏆 WINNER: {model2_name} (+{score2-score1:.3f})")
    else:
        print(f"\n🤝 TIE")
    
    print('='*80 + "\n")


if __name__ == "__main__":
    main()