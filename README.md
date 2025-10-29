# Divide-and-Conquer
Hierarchical subproblem based RL for process verifiable step rewards enhancing LLM reasoning models

# To run batch decompose and evals on decomposed subproblems
python batch_decompose.py --models hf:Qwen/Qwen3-4B-Instruct-2507 --num-problems 50

# To only run evals on decomposed subproblems already stored 
python score_existing.py --dataset-dir data/decompositions --output scores_report.json --threshold 0.7 

