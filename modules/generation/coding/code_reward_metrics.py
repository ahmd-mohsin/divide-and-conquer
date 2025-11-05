# code_reward_metrics.py
# -*- coding: utf-8 -*-
"""
Code reward metrics with safe fallbacks.

- Fixes NameError for typing.Dict / typing.Any by importing from typing.
- Provides a self-contained CodeRewardCalculator with:
    * _extract_code_from_chain
    * compute_codebleu (tries real CodeBLEU; falls back gracefully)
    * compute_ast_similarity
    * compute_codebert_score (tries Transformers; falls back gracefully)
    * compute_reward (weighted combination)
- No hard dependency on external libraries: optional imports are tried; otherwise
  simple, deterministic fallbacks are used so training/evaluation can proceed.

Author: ICML 2026 project
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, List, Tuple
import ast
import math
import re

# ------------------------------- Utilities ----------------------------------


def _normalize_ws(text: str) -> str:
    """Collapse whitespace to single spaces for simple token-based comparisons."""
    return re.sub(r"\s+", " ", text.strip())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -------------------------- Main Calculator Class ---------------------------


class CodeRewardCalculator:
    """
    Compute code-level rewards using a mixture of metrics with robust fallbacks.

    Parameters
    ----------
    use_codebleu : bool
        If True, attempts to compute CodeBLEU (with fallback).
    use_ast_similarity : bool
        If True, computes AST Jaccard similarity (with token fallback).
    use_codebert : bool
        If True, attempts to compute CodeBERT cosine similarity (with fallback).
    success_threshold : float
        Threshold for considering a sample successful (0..1).
    weights : Dict[str, float]
        Weights for the metric combination. Keys: 'codebleu', 'ast', 'codebert'.
    verbose : bool
        If True, prints warnings when optional backends fail.
    """

    def __init__(
        self,
        use_codebleu: bool = True,
        use_ast_similarity: bool = True,
        use_codebert: bool = True,
        success_threshold: float = 0.7,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> None:
        self.use_codebleu = use_codebleu
        self.use_ast_similarity = use_ast_similarity
        self.use_codebert = use_codebert
        self.success_threshold = float(success_threshold)
        self.verbose = verbose

        self.weights: Dict[str, float] = {
            "codebleu": 1.0,
            "ast": 1.0,
            "codebert": 1.0,
        }
        if weights:
            self.weights.update(weights)

        # Lazy loaders / caches
        self._codebleu_fn = None  # callable or None
        self._codebert_pipeline = None  # (tokenizer, model, device) or None

    # ---------------------------- Code Extraction ----------------------------

    def _extract_code_from_chain(self, chain: str) -> str:
        """
        Extract code from a reasoning chain. Tries several strategies.
        """
        # Method 1: Markdown fenced blocks
        pattern = r"```(?:python)?\s*(.*?)```"
        matches = re.findall(pattern, chain, re.DOTALL | re.IGNORECASE)
        if matches:
            code = matches[-1].strip()
            if code and len(code) > 10:
                return code

        # Method 2: <code> tags
        code_tag_pattern = r"<code>(.*?)</code>"
        matches = re.findall(code_tag_pattern, chain, re.DOTALL | re.IGNORECASE)
        if matches:
            code = matches[-1].strip()
            if code and len(code) > 10:
                return code

        # Method 3: function definitions
        func_pattern = r"(def\s+\w+\(.*?\):.*?)(?=\n\S|\Z)"
        matches = re.findall(func_pattern, chain, re.DOTALL)
        if matches:
            return "\n".join(matches)

        # Method 4: imports/defs/classes heuristic
        if "import" in chain.lower() or "from" in chain.lower():
            lines = chain.split("\n")
            code_lines: List[str] = []
            in_code = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ", "def ", "class ")):
                    in_code = True
                if in_code:
                    # stop if clearly prose (heuristic)
                    if (
                        stripped
                        and not any(
                            stripped.startswith(s)
                            for s in ("import", "from", "def", "class", "#", " ", "\t")
                        )
                        and "=" not in stripped
                    ):
                        if len(stripped) > 50 and " " in stripped:
                            break
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()
                if len(code) > 10:
                    return code

        # Method 5: last significant paragraph
        parts = chain.split("\n\n")
        if len(parts) > 1:
            for part in reversed(parts):
                if len(part.strip()) > 20:
                    return part.strip()

        # Last resort
        return chain.strip()

    # ---------------------------- CodeBLEU (safe) ----------------------------

    def _load_codebleu(self):
        """
        Try to load a CodeBLEU calculator. If unavailable, return a simple
        n-gram overlap fallback that imitates the interface.

        Returns
        -------
        callable or None
            A function f(references=[[ref]], predictions=[pred], lang=..., weights=..., tokenizer=None)
            that returns {'codebleu': float}, or None on total failure.
        """
        if self._codebleu_fn is not None:
            return self._codebleu_fn

        # Try the real implementation
        try:
            # Common third-party package API
            #   from codebleu import calc_codebleu
            # not always present; wrap if found
            from codebleu import calc_codebleu  # type: ignore

            def _real_cb(**kwargs):
                return calc_codebleu(**kwargs)

            self._codebleu_fn = _real_cb
            return self._codebleu_fn
        except Exception as e:
            if self.verbose:
                print(f"[CodeBLEU] real implementation not available: {e}")

        # Fallback: simple n-gram overlap across n=1..4 (BLEU-like, no brevity penalty)
        def _fallback_codebleu(
            references,
            predictions,
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        ):
            try:
                ref = references[0][0]
                pred = predictions[0]
            except Exception:
                return {"codebleu": 0.0}

            ref_toks = _normalize_ws(ref).split()
            pred_toks = _normalize_ws(pred).split()
            if not ref_toks or not pred_toks:
                return {"codebleu": 0.0}

            precisions: List[float] = []
            for n, w in zip((1, 2, 3, 4), weights):
                if w == 0:
                    precisions.append(0.0)
                    continue
                ref_ngrams = _ngrams(ref_toks, n)
                pred_ngrams = _ngrams(pred_toks, n)
                if not ref_ngrams or not pred_ngrams:
                    precisions.append(0.0)
                    continue
                ref_set = set(ref_ngrams)
                pred_set = set(pred_ngrams)
                score = len(ref_set & pred_set) / max(1, len(pred_set))
                precisions.append(score)

            # Geometric mean of precisions (avoid nan)
            try:
                gm = math.exp(
                    sum(math.log(max(p, 1e-12)) * w for p, w in zip(precisions, weights))
                )
            except Exception:
                gm = 0.0

            return {"codebleu": float(max(0.0, min(1.0, gm)))}

        self._codebleu_fn = _fallback_codebleu
        return self._codebleu_fn

    def compute_codebleu(self, predicted: str, reference: str, lang: str = "python") -> float:
        """
        Compute CodeBLEU (or fallback) with defensive guards.
        """
        if not self.use_codebleu:
            return 0.0
        if not predicted or not reference or len(predicted) < 5 or len(reference) < 5:
            return 0.0

        try:
            calc_codebleu = self._load_codebleu()
            if calc_codebleu is None:
                return 0.0

            predicted_clean = predicted.strip()
            reference_clean = reference.strip()

            result = calc_codebleu(
                references=[[reference_clean]],
                predictions=[predicted_clean],
                lang=lang,
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=None,
            )
            return float(result.get("codebleu", 0.0))
        except TypeError as e:
            # Known oddities in some CodeBLEU forks
            if "integer is required" in str(e).lower():
                pred_tokens = set(predicted.split())
                ref_tokens = set(reference.split())
                if not pred_tokens or not ref_tokens:
                    return 0.0
                return len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
            if self.verbose:
                print(f"CodeBLEU TypeError: {e}")
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"CodeBLEU error: {e}")
            return 0.0

    # -------------------------- AST Similarity (safe) ------------------------

    def _extract_ast_nodes(self, tree: ast.AST) -> Set[str]:
        """Collect node type names in an AST as a set."""
        names: Set[str] = set()
        for node in ast.walk(tree):
            names.add(type(node).__name__)
        return names

    def compute_ast_similarity(self, predicted: str, reference: str) -> float:
        """
        Compute Jaccard similarity over AST node-type sets. Falls back to token
        overlap when either side fails to parse.
        """
        if not self.use_ast_similarity:
            return 0.0
        if not predicted or not reference or len(predicted) < 5 or len(reference) < 5:
            return 0.0

        try:
            pred_tree = ast.parse(predicted)
            ref_tree = ast.parse(reference)

            pred_nodes = self._extract_ast_nodes(pred_tree)
            ref_nodes = self._extract_ast_nodes(ref_tree)

            if not pred_nodes and not ref_nodes:
                return 1.0
            if not pred_nodes or not ref_nodes:
                return 0.0

            inter = len(pred_nodes & ref_nodes)
            union = len(pred_nodes | ref_nodes)
            return inter / union if union else 0.0

        except SyntaxError:
            # Token fallback (with penalty)
            pred_clean = "\n".join(
                line for line in predicted.split("\n")
                if line.strip() and not line.strip().startswith("#")
            )
            ref_clean = "\n".join(
                line for line in reference.split("\n")
                if line.strip() and not line.strip().startswith("#")
            )
            pred_tokens = set(pred_clean.split())
            ref_tokens = set(ref_clean.split())
            if not pred_tokens or not ref_tokens:
                return 0.0
            overlap = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
            return overlap * 0.8
        except Exception as e:
            if self.verbose:
                print(f"AST error: {e}")
            return 0.0

    # -------------------------- CodeBERT Similarity --------------------------

    def _load_codebert(self):
        """
        Attempt to load CodeBERT (microsoft/codebert-base). Returns a tuple
        (tokenizer, model, device) or None if unavailable.
        """
        if self._codebert_pipeline is not None:
            return self._codebert_pipeline

        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModel  # type: ignore

            model_name = "microsoft/codebert-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self._codebert_pipeline = (tokenizer, model, device)
            return self._codebert_pipeline
        except Exception as e:
            if self.verbose:
                print(f"[CodeBERT] not available: {e}")
            self._codebert_pipeline = None
            return None

    def _cosine_sim(self, a, b) -> float:
        try:
            import torch  # type: ignore

            a = a / (torch.norm(a) + 1e-12)
            b = b / (torch.norm(b) + 1e-12)
            return float(torch.clamp((a * b).sum(), -1.0, 1.0).item())
        except Exception:
            return 0.0

    def compute_codebert_score(self, predicted: str, reference: str) -> float:
        """
        Compute cosine similarity between [CLS] embeddings from CodeBERT.
        Falls back to normalized token overlap if transformers/torch are absent.
        """
        if not self.use_codebert:
            return 0.0
        if not predicted or not reference or len(predicted) < 5 or len(reference) < 5:
            return 0.0

        pipe = self._load_codebert()
        if pipe is None:
            # Simple fallback: token Jaccard (scaled to [0,1])
            pred_tokens = set(_normalize_ws(predicted).split())
            ref_tokens = set(_normalize_ws(reference).split())
            return _jaccard(pred_tokens, ref_tokens)

        try:
            import torch  # type: ignore

            tokenizer, model, device = pipe
            with torch.no_grad():
                enc_pred = tokenizer(
                    predicted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                enc_ref = tokenizer(
                    reference,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                # Use the last hidden state CLS token representation
                pred_out = model(**enc_pred)[0][:, 0, :].squeeze(0)  # (hidden,)
                ref_out = model(**enc_ref)[0][:, 0, :].squeeze(0)    # (hidden,)
                return max(0.0, min(1.0, (self._cosine_sim(pred_out, ref_out) + 1) / 2))
        except Exception as e:
            if self.verbose:
                print(f"CodeBERT error: {e}")
            # Fallback again
            pred_tokens = set(_normalize_ws(predicted).split())
            ref_tokens = set(_normalize_ws(reference).split())
            return _jaccard(pred_tokens, ref_tokens)

    # --------------------------- Combined Reward -----------------------------

    def compute_reward(
        self,
        predicted_code: str,
        reference_code: str,
        lang: str = "python",
    ) -> Dict[str, Any]:
        """
        Compute combined reward using enabled metrics with defensive extraction.
        """
        # Extract code if chain reasoning likely
        if ("Step 1:" in predicted_code) or ("```" in predicted_code):
            predicted_code = self._extract_code_from_chain(predicted_code)

        if not predicted_code or len(predicted_code) < 5:
            return {
                "codebleu": 0.0,
                "ast_similarity": 0.0,
                "codebert_score": 0.0,
                "final_reward": 0.0,
                "is_successful": False,
                "threshold": self.success_threshold,
                "error": "No code extracted",
            }

        scores: Dict[str, float] = {}

        if self.use_codebleu:
            scores["codebleu"] = self.compute_codebleu(predicted_code, reference_code, lang)
        else:
            scores["codebleu"] = 0.0

        if self.use_ast_similarity:
            scores["ast_similarity"] = self.compute_ast_similarity(predicted_code, reference_code)
        else:
            scores["ast_similarity"] = 0.0

        if self.use_codebert:
            scores["codebert_score"] = self.compute_codebert_score(predicted_code, reference_code)
        else:
            scores["codebert_score"] = 0.0

        active_metrics = int(self.use_codebleu) + int(self.use_ast_similarity) + int(self.use_codebert)
        if active_metrics == 0:
            final_reward = 0.0
        else:
            total_weight = (
                (self.weights.get("codebleu", 0.0) if self.use_codebleu else 0.0)
                + (self.weights.get("ast", 0.0) if self.use_ast_similarity else 0.0)
                + (self.weights.get("codebert", 0.0) if self.use_codebert else 0.0)
            )
            if total_weight > 0.0:
                final_reward = (
                    scores["codebleu"] * (self.weights.get("codebleu", 0.0) if self.use_codebleu else 0.0)
                    + scores["ast_similarity"] * (self.weights.get("ast", 0.0) if self.use_ast_similarity else 0.0)
                    + scores["codebert_score"] * (self.weights.get("codebert", 0.0) if self.use_codebert else 0.0)
                ) / total_weight
            else:
                final_reward = 0.0

        is_successful = final_reward >= self.success_threshold

        return {
            "codebleu": scores["codebleu"],
            "ast_similarity": scores["ast_similarity"],
            "codebert_score": scores["codebert_score"],
            "final_reward": final_reward,
            "is_successful": is_successful,
            "threshold": self.success_threshold,
        }


__all__ = ["CodeRewardCalculator"]
