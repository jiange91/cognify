leave = 'inner'
dir = "/mnt/data/ssd2/zijian/cognify/examples/text-to-sql/b64_l2_4_14/outer/outer_trial_1"

from pathlib import Path
folder_path = Path(dir)

# Recursively find all .json files
json_files = list(folder_path.rglob('opt_logs.json'))

import json

# Print the found files
trace = []
for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        for d in data:
            if data[d]["layer_name"] != leave:
                break
            result = data[d]["result"]['summary']
            # trace.append((result['reduced_score'], result['reduced_price'] * 1000))
            trace.append((result['reduced_score'], result['reduced_exec_time']))
            

import numpy as np
def get_pareto_front(candidates: list):
    if not candidates:
        return []

    vectors = np.array([[-score, price] for score, price in candidates])
    is_efficient = np.ones(vectors.shape[0], dtype=bool)
    for i, v in enumerate(vectors):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                vectors[is_efficient] < v, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    # return filtered [T_ParetoProgram]
    pareto_frontier = [
        log for log, eff in zip(candidates, is_efficient) if eff
    ]
    return pareto_frontier
    
print(len(trace))
pareto = get_pareto_front(trace)
print(pareto)

import os
with open(os.path.join(dir, 'trace.json'), 'w') as f:
    json.dump(trace, f, indent=4)
    
    