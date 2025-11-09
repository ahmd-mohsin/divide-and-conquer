#!/usr/bin/env python3
"""
Decomposition Quality Evaluation
Compare decompositions from two models using three metric categories.

Place in: ~/Divide-and-Conquer/data/evaluation/
Run from: ~/Divide-and-Conquer/data/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
class ComparisonResult:
    """Comparison for one problem"""
    problem_id: str
    problem_text: str
    model1_name: str
    model2_name: str
    model1_subproblems: int
    model2_subproblems: int
    model1_dag: DAGMetrics
    model2_dag: DAGMetrics
    model1_faith: FaithfulnessMetrics
    model2_faith: FaithfulnessMetrics
    model1_lfc: LFCMetrics
    model2_lfc: LFCMetrics
    model1_overall: float
    model2_overall: float
    winner: str


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
        step_size_cv = np.std(step_sizes) / max(avg_step_size, 1)
        
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


class DecompositionComparator:
    """Compare two decompositions"""
    
    def __init__(self):
        self.analyzer = DecompositionAnalyzer()
    
    def load_dataset(self, chains_dir: Path) -> Dict[str, Dict]:
        """Load all problems from chains directory"""
        problems = {}
        for problem_file in sorted(chains_dir.glob("problem_*.json")):
            with open(problem_file, 'r') as f:
                data = json.load(f)
                problems[data.get('problem', '')] = data
        return problems
    
    def find_matching_problems(
        self, dataset1: Dict[str, Dict], dataset2: Dict[str, Dict], max_problems: int = 100
    ) -> List[Tuple[str, Dict, Dict]]:
        """Find problems in both datasets"""
        common = set(dataset1.keys()) & set(dataset2.keys())
        return [(p, dataset1[p], dataset2[p]) for p in sorted(common)[:max_problems]]
    
    def compare_problem(
        self, problem_text: str, data1: Dict, data2: Dict,
        model1_name: str, model2_name: str
    ) -> ComparisonResult:
        """Compare two decompositions"""
        G1 = self.analyzer.build_dag(data1)
        G2 = self.analyzer.build_dag(data2)
        
        dag1 = self.analyzer.compute_dag_metrics(G1, data1)
        dag2 = self.analyzer.compute_dag_metrics(G2, data2)
        faith1 = self.analyzer.compute_faithfulness_metrics(G1, data1)
        faith2 = self.analyzer.compute_faithfulness_metrics(G2, data2)
        lfc1 = self.analyzer.compute_lfc_metrics(G1, data1)
        lfc2 = self.analyzer.compute_lfc_metrics(G2, data2)
        
        score1 = self._compute_overall_score(dag1, faith1, lfc1)
        score2 = self._compute_overall_score(dag2, faith2, lfc2)
        
        winner = model1_name if score1 > score2 else (model2_name if score2 > score1 else "Tie")
        
        return ComparisonResult(
            data1.get('decomposition_id', 'unknown'), problem_text[:100] + "...",
            model1_name, model2_name,
            data1.get('num_subproblems', 0), data2.get('num_subproblems', 0),
            dag1, dag2, faith1, faith2, lfc1, lfc2,
            score1, score2, winner
        )
    
    def _compute_overall_score(self, dag: DAGMetrics, faith: FaithfulnessMetrics, lfc: LFCMetrics) -> float:
        """Compute overall quality score"""
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


def aggregate_results(results: List[ComparisonResult]) -> Dict:
    """Aggregate comparison results"""
    if not results:
        return {}
    
    model1_name = results[0].model1_name
    model2_name = results[0].model2_name
    
    model1_wins = sum(1 for r in results if r.winner == model1_name)
    model2_wins = sum(1 for r in results if r.winner == model2_name)
    ties = sum(1 for r in results if r.winner == "Tie")
    
    def avg_attr(attr_path: str) -> float:
        values = []
        for r in results:
            obj = r
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, bool):
                obj = 1.0 if obj else 0.0
            values.append(float(obj))
        return np.mean(values)
    
    return {
        "total_problems": len(results),
        "wins": {model1_name: model1_wins, model2_name: model2_wins, "ties": ties},
        "win_rates": {model1_name: model1_wins / len(results), model2_name: model2_wins / len(results)},
        "avg_subproblems": {model1_name: avg_attr("model1_subproblems"), model2_name: avg_attr("model2_subproblems")},
        "dag_metrics": {
            "avg_depth": {model1_name: avg_attr("model1_dag.depth"), model2_name: avg_attr("model2_dag.depth")},
            "avg_parallelism": {model1_name: avg_attr("model1_dag.parallelism"), model2_name: avg_attr("model2_dag.parallelism")},
            "avg_justified_edge_rate": {model1_name: avg_attr("model1_dag.justified_edge_rate"), model2_name: avg_attr("model2_dag.justified_edge_rate")},
            "avg_duplicate_goal_rate": {model1_name: avg_attr("model1_dag.duplicate_goal_rate"), model2_name: avg_attr("model2_dag.duplicate_goal_rate")}
        },
        "faithfulness_metrics": {
            "avg_redundancy_rate": {model1_name: avg_attr("model1_faith.redundancy_rate"), model2_name: avg_attr("model2_faith.redundancy_rate")},
            "avg_dependency_usage": {model1_name: avg_attr("model1_faith.avg_dependency_usage"), model2_name: avg_attr("model2_faith.avg_dependency_usage")},
            "avg_specificity": {model1_name: avg_attr("model1_faith.specificity_score"), model2_name: avg_attr("model2_faith.specificity_score")}
        },
        "lfc_metrics": {
            "avg_type_consistency": {model1_name: avg_attr("model1_lfc.type_consistency"), model2_name: avg_attr("model2_lfc.type_consistency")},
            "avg_skill_diversity": {model1_name: avg_attr("model1_lfc.skill_diversity"), model2_name: avg_attr("model2_lfc.skill_diversity")},
            "avg_lfc_score": {model1_name: avg_attr("model1_lfc.lfc_score"), model2_name: avg_attr("model2_lfc.lfc_score")}
        },
        "overall_scores": {model1_name: avg_attr("model1_overall"), model2_name: avg_attr("model2_overall")}
    }


def print_summary(aggregate: Dict):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("DECOMPOSITION QUALITY COMPARISON")
    print("="*80)
    
    model1_name = list(aggregate["wins"].keys())[0]
    model2_name = list(aggregate["wins"].keys())[1]
    
    print(f"\nProblems: {aggregate['total_problems']}")
    print(f"\nWins:")
    print(f"  {model1_name}: {aggregate['wins'][model1_name]} ({aggregate['win_rates'][model1_name]:.1%})")
    print(f"  {model2_name}: {aggregate['wins'][model2_name]} ({aggregate['win_rates'][model2_name]:.1%})")
    print(f"  Ties: {aggregate['wins']['ties']}")
    
    print(f"\nAvg Subproblems:")
    print(f"  {model1_name}: {aggregate['avg_subproblems'][model1_name]:.1f}")
    print(f"  {model2_name}: {aggregate['avg_subproblems'][model2_name]:.1f}")
    
    print(f"\n--- DAG Structural Quality ---")
    print(f"Depth: {model1_name}={aggregate['dag_metrics']['avg_depth'][model1_name]:.2f}, {model2_name}={aggregate['dag_metrics']['avg_depth'][model2_name]:.2f}")
    print(f"Parallelism: {model1_name}={aggregate['dag_metrics']['avg_parallelism'][model1_name]:.2f}, {model2_name}={aggregate['dag_metrics']['avg_parallelism'][model2_name]:.2f}")
    print(f"Edge Justification: {model1_name}={aggregate['dag_metrics']['avg_justified_edge_rate'][model1_name]:.2%}, {model2_name}={aggregate['dag_metrics']['avg_justified_edge_rate'][model2_name]:.2%}")
    
    print(f"\n--- Faithfulness & Minimality ---")
    print(f"Redundancy: {model1_name}={aggregate['faithfulness_metrics']['avg_redundancy_rate'][model1_name]:.2%}, {model2_name}={aggregate['faithfulness_metrics']['avg_redundancy_rate'][model2_name]:.2%}")
    print(f"Dependency Usage: {model1_name}={aggregate['faithfulness_metrics']['avg_dependency_usage'][model1_name]:.2%}, {model2_name}={aggregate['faithfulness_metrics']['avg_dependency_usage'][model2_name]:.2%}")
    print(f"Specificity: {model1_name}={aggregate['faithfulness_metrics']['avg_specificity'][model1_name]:.2%}, {model2_name}={aggregate['faithfulness_metrics']['avg_specificity'][model2_name]:.2%}")
    
    print(f"\n--- Compositional Skill (LFC) ---")
    print(f"Type Consistency: {model1_name}={aggregate['lfc_metrics']['avg_type_consistency'][model1_name]:.2%}, {model2_name}={aggregate['lfc_metrics']['avg_type_consistency'][model2_name]:.2%}")
    print(f"Skill Diversity: {model1_name}={aggregate['lfc_metrics']['avg_skill_diversity'][model1_name]:.2%}, {model2_name}={aggregate['lfc_metrics']['avg_skill_diversity'][model2_name]:.2%}")
    
    print(f"\n--- Overall Scores ---")
    print(f"{model1_name}: {aggregate['overall_scores'][model1_name]:.3f}")
    print(f"{model2_name}: {aggregate['overall_scores'][model2_name]:.3f}")
    
    winner = model1_name if aggregate['win_rates'][model1_name] > aggregate['win_rates'][model2_name] else model2_name
    print(f"\n🏆 WINNER: {winner}")
    print("="*80 + "\n")


def main():
    # Configuration
    chains_dir1 = Path("/home/ahmed/Divide-and-Conquer/data/chains_calc_gsm8k/chains")
    chains_dir2 = Path("/home/ahmed/Divide-and-Conquer/data/gsm8k_deepseek_r1/chains")
    model1_name = "Qwen-7B"
    model2_name = "DeepSeek-R1-14B"
    max_problems = 100
    
    print("="*80)
    print("DECOMPOSITION EVALUATION")
    print("="*80)
    print(f"\nModel 1: {model1_name} ({chains_dir1})")
    print(f"Model 2: {model2_name} ({chains_dir2})")
    print(f"Max problems: {max_problems}\n")
    
    # Load and compare
    comparator = DecompositionComparator()
    dataset1 = comparator.load_dataset(chains_dir1)
    dataset2 = comparator.load_dataset(chains_dir2)
    
    print(f"Loaded: {model1_name}={len(dataset1)}, {model2_name}={len(dataset2)}")
    
    matches = comparator.find_matching_problems(dataset1, dataset2, max_problems)
    print(f"Matching problems: {len(matches)}\n")
    
    if not matches:
        print("ERROR: No matching problems!")
        return
    
    print("Comparing decompositions...")
    results = []
    for i, (problem_text, data1, data2) in enumerate(matches, 1):
        print(f"  [{i}/{len(matches)}] {problem_text[:60]}...")
        results.append(comparator.compare_problem(problem_text, data1, data2, model1_name, model2_name))
    
    # Save results
    aggregate = aggregate_results(results)
    
    output_file = Path("evaluation/results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"summary": aggregate, "detailed": [asdict(r) for r in results]}, f, indent=2)
    
    print(f"\n✓ Saved: {output_file}")
    
    # Print summary
    print_summary(aggregate)


if __name__ == "__main__":
    main()