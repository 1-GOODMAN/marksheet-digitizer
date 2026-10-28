"""
Page-1-Only Z3 MaxSMT Solver for Marksheet Digitizer.
Enforces all arithmetic invariants derivable from Page 1 (CIE table) alone.
Decision variables: test1, test2, aba, theory_total, viva, lab, pract_total, cie_total.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import z3


class MarksheetSolver:
    """
    Z3 MaxSMT optimizer for reconciling Page-1 OCR predictions against
    hard institutional arithmetic constraints.

    Hard Constraints (from Page 1 CIE table):
      1. test1 + test2 + aba == theory_total
      2. viva + lab == pract_total
      3. theory_total + pract_total == cie_total

    Variable Domains:
      test1, test2  : [0, 15]
      aba           : [0, 10]
      theory_total  : [0, 40]
      viva, lab     : [0, 5]
      pract_total   : [0, 10]
      cie_total     : [0, 50]
    """

    # Canonical Page-1 variable names (same keys used by preprocessor & pipeline)
    P1_VAR_NAMES = [
        "test1", "test2", "aba",
        "theory_total", "viva", "lab",
        "pract_total", "cie_total",
    ]

    def __init__(self):
        pass

    def solve(
        self,
        candidate_distributions: Dict[str, Dict[int, float]],
        fixed_values: Optional[Dict[str, int]] = None,
        crop_stds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Maximises total log-likelihood subject to the three hard constraints.
        Enforces Total-Anchored Confidence Multipliers and Algebraic Missing-Cell Inference:
        1. Two-digit summary totals (cie_total in 25..50, theory_total in 18..40) get a
           3.0x confidence multiplier when visual OCR probability > 0.50.
        2. Penalizes value 0 on test1, test2, aba, viva, lab whenever crop_stds > 10.0.
        3. If theory_total and two of (test1, test2, aba) are confident, algebraically infers
           the missing third cell so valid totals are never corrupted.
        4. If pract_total is clear, algebraically infers viva/lab within 0..5 to match pract_total.
        """
        opt = z3.Optimize()

        # Copy candidate distributions so we can inject algebraic candidates safely
        dists = {k: dict(v) for k, v in candidate_distributions.items()}
        stds = crop_stds or {}

        # --- 0. Algebraic Candidate Inference ---
        # A. Theory breakdown: test1 + test2 + aba == theory_total
        th_cands = [val for val, p in dists.get("theory_total", {}).items() if 18 <= val <= 40 and p >= 0.35]
        t1_cands = [val for val, p in dists.get("test1", {}).items() if 1 <= val <= 15 and p >= 0.40]
        t2_cands = [val for val, p in dists.get("test2", {}).items() if 1 <= val <= 15 and p >= 0.40]
        aba_cands = [val for val, p in dists.get("aba", {}).items() if 1 <= val <= 10 and p >= 0.40]

        for th_v in th_cands:
            # Case 1: test1 & aba known -> infer test2
            for t1_v in t1_cands:
                for aba_v in aba_cands:
                    inferred_t2 = th_v - t1_v - aba_v
                    if 0 <= inferred_t2 <= 15:
                        dists.setdefault("test2", {})
                        dists["test2"][inferred_t2] = max(dists["test2"].get(inferred_t2, 0.0), 0.75)
            # Case 2: test2 & aba known -> infer test1
            for t2_v in t2_cands:
                for aba_v in aba_cands:
                    inferred_t1 = th_v - t2_v - aba_v
                    if 0 <= inferred_t1 <= 15:
                        dists.setdefault("test1", {})
                        dists["test1"][inferred_t1] = max(dists["test1"].get(inferred_t1, 0.0), 0.75)
            # Case 3: test1 & test2 known -> infer aba
            for t1_v in t1_cands:
                for t2_v in t2_cands:
                    inferred_aba = th_v - t1_v - t2_v
                    if 0 <= inferred_aba <= 10:
                        dists.setdefault("aba", {})
                        dists["aba"][inferred_aba] = max(dists["aba"].get(inferred_aba, 0.0), 0.75)

        # B. Practical breakdown: viva + lab == pract_total
        pr_cands = [val for val, p in dists.get("pract_total", {}).items() if 5 <= val <= 10 and p >= 0.40]
        viva_cands = [val for val, p in dists.get("viva", {}).items() if 1 <= val <= 5 and p >= 0.40]
        lab_cands = [val for val, p in dists.get("lab", {}).items() if 1 <= val <= 5 and p >= 0.40]

        for pr_v in pr_cands:
            for l_v in lab_cands:
                inferred_v = pr_v - l_v
                if 0 <= inferred_v <= 5:
                    dists.setdefault("viva", {})
                    dists["viva"][inferred_v] = max(dists["viva"].get(inferred_v, 0.0), 0.75)
            for v_v in viva_cands:
                inferred_l = pr_v - v_v
                if 0 <= inferred_l <= 5:
                    dists.setdefault("lab", {})
                    dists["lab"][inferred_l] = max(dists["lab"].get(inferred_l, 0.0), 0.75)

        # C. Total CIE reconciliation: theory_total + pract_total == cie_total
        cie_cands = [val for val, p in dists.get("cie_total", {}).items() if 20 <= val <= 50 and p >= 0.40]
        for cie_v in cie_cands:
            for pr_v in pr_cands:
                inferred_th = cie_v - pr_v
                if 0 <= inferred_th <= 40:
                    dists.setdefault("theory_total", {})
                    dists["theory_total"][inferred_th] = max(dists["theory_total"].get(inferred_th, 0.0), 0.85)

        # --- 1. Declare Z3 integer variables ---
        t1   = z3.Int("test1")
        t2   = z3.Int("test2")
        aba  = z3.Int("aba")
        th   = z3.Int("theory_total")
        viva = z3.Int("viva")
        lab  = z3.Int("lab")
        pr   = z3.Int("pract_total")
        cie  = z3.Int("cie_total")

        all_vars: Dict[str, z3.ArithRef] = {
            "test1":        t1,
            "test2":        t2,
            "aba":          aba,
            "theory_total": th,
            "viva":         viva,
            "lab":          lab,
            "pract_total":  pr,
            "cie_total":    cie,
        }

        # --- 2. Domain bounds ---
        opt.add(t1  >= 0, t1  <= 15)
        opt.add(t2  >= 0, t2  <= 15)
        opt.add(aba >= 0, aba <= 10)
        opt.add(th  >= 0, th  <= 40)
        opt.add(viva >= 0, viva <= 5)
        opt.add(lab  >= 0, lab  <= 5)
        opt.add(pr  >= 0, pr  <= 10)
        opt.add(cie >= 0, cie <= 50)

        # --- 3. Hard arithmetic constraints (Page 1 CIE table) ---
        opt.add(t1 + t2 + aba == th)     # C1: Theory breakdown
        opt.add(viva + lab == pr)         # C2: Practical breakdown
        opt.add(th + pr == cie)           # C3: Total CIE

        # --- 4. Fixed / ground-truth overrides ---
        if fixed_values:
            for k, val in fixed_values.items():
                if k in all_vars:
                    opt.add(all_vars[k] == int(val))

        # --- 5. Log-likelihood soft objective ---
        scale_factor = 1000
        min_log_p = -10.0

        for var_name, dist in dists.items():
            if var_name not in all_vars or not dist:
                continue
            z_var = all_vars[var_name]
            total_p = sum(dist.values())
            if total_p <= 0:
                continue
            for val, prob in dist.items():
                p = max(prob / total_p, 1e-4)
                log_p = float(np.log(p))
                weight = max(1, int((log_p - min_log_p) * scale_factor))

                # Fix 3.1: 3.0x Confidence Multiplier for two-digit summary totals
                if var_name == "cie_total" and 25 <= val <= 50 and p >= 0.40:
                    weight = int(weight * 3.5)
                elif var_name == "theory_total" and 18 <= val <= 40 and p >= 0.40:
                    weight = int(weight * 3.0)

                # Fix 3.2: Penalize value 0 strongly on inked cells
                if val == 0 and var_name in ["test1", "test2", "aba", "viva", "lab"]:
                    if stds.get(var_name, 0.0) > 10.0:
                        weight = max(1, int(weight * 0.08))

                opt.add_soft(z_var == val, weight=weight)

        # --- 6. Solve and extract ---
        check_res = opt.check()
        if check_res != z3.sat:
            return {
                "success": False,
                "values": {},
                "log_likelihood": float("-inf"),
                "reason": "Unsatisfiable constraints",
            }

        model = opt.model()
        resolved: Dict[str, int] = {}
        for var_name, z_var in all_vars.items():
            val = model.eval(z_var)
            if val is not None:
                resolved[var_name] = val.as_long()

        # Compute achieved log-likelihood
        total_ll = 0.0
        for var_name, dist in candidate_distributions.items():
            if var_name in resolved and resolved[var_name] in dist:
                total_p = sum(dist.values())
                p = max(dist[resolved[var_name]] / total_p, 1e-6)
                total_ll += float(np.log(p))

        return {
            "success": True,
            "values": resolved,
            "log_likelihood": total_ll,
        }
