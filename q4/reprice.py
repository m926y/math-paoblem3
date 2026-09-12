"""Reprice the saved Q2/Q3 actions without modifying or rerunning their policies."""
from pathlib import Path
import numpy as np
from .runner import inputs
from microgrid.reporting import save_json

def main():
    root=Path(__file__).resolve().parents[1]
    _,prices,_=inputs(root);p=prices.actual[31:];result={}
    for q in (2,3):
        with np.load(root/f'output/q{q}/trace.npz',allow_pickle=False) as a:
            g=a['grid'][31:];g0=a['grid0'][31:];e=a['emergency'][31:]
        normal=float(np.sum(p*g));urgent=float(5*np.sum(p*e));adjust=float(.5*np.sum(p*np.abs(g-g0))) if q==3 else 0.
        result[str(q)]=dict(normal_cost_cny=normal,adjustment_cost_cny=adjust,emergency_cost_cny=urgent,total_cost_cny=normal+adjust+urgent,
                            note='Original fixed-price actions repriced only; not reoptimized under new prices')
    save_json(root/'output/q4_experiments/original_policy_repriced.json',result)

if __name__=='__main__':main()
