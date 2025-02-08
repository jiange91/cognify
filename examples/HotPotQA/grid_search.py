import cognify
from cognify.llm.model import LMConfig
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.llm import Demonstration, Model
from grid_few_shot import agent_few_shots 
import copy

# structure_params = [NoChange(), ensemble.UniversalSelfConsistency(3)]

# used for hashing
old_model = 'fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-415aeb86'

model_selection_params = []
for agent_name in agent_few_shots:
    ms = model_selection.model_selection_factory(
        [
            LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-62baf9fc'),
            LMConfig(model='gpt-4o-mini'),
        ]
    )
    ms.module_name = agent_name
    model_selection_params.append(ms)

reasoning_params = []
for agent_name in agent_few_shots:
    rp = reasoning.LMReasoning([NoChange(), reasoning.ZeroShotCoT()])
    rp.module_name = agent_name
    reasoning_params.append(rp)
    
search_space = {}
cog_space = {}
for fewshot in agent_few_shots.values():
    search_space[fewshot.hash] = list(fewshot.options.keys())
    cog_space[fewshot.hash] = fewshot
for ms in model_selection_params:
    search_space[ms.hash] = list(ms.options.keys())
    cog_space[ms.hash] = ms
for rp in reasoning_params:
    search_space[rp.hash] = list(rp.options.keys())
    cog_space[rp.hash] = rp
    
import json

# with open("grid_results/search_space.json", "w") as f:
#     json.dump(search_space, f, indent=4)

import itertools
import sys
import hashlib
from cognify.optimizer.plugin import OptimizerSchema, capture_module_from_fs
from cognify.optimizer.core.flow import EvaluationResult
from cognify.graph.program import Workflow, Module
from config import load_data_minor, answer_f1

_, _, devset = load_data_minor()

batch = 50
n_parallel = 16

import multiprocessing as mp
import time

def run_once(tid, input, label, config, cog_space, sema, q):
    # load the module
    sys.argv = ["workflow.py"]
    schema = OptimizerSchema.capture("workflow.py")
    module_pool = {m.name: m for m in schema.opt_target_modules}
   
    # apply config
    for cog_name, cog_option in config.items():
        cog = cog_space[cog_name]
        cog.apply_option(cog_option, module_pool[cog.module_name])
    for m in module_pool.values():
        m.reset()
        
    sema.acquire()
    
    try:
        score, price, exec_time = 0.0, 0.0, 0.0
        result = None
        
        start_time = time.time()
        result = schema.program(**input)
        exec_time = time.time() - start_time
        
        score = answer_f1(answer=result['answer'], ground_truth=label['ground_truth'])
        price = 0.0
        for lm in Module.all_of_type(module_pool.values(), Model):
            price += lm.get_total_cost()
    except Exception as e:
        # catch any errors thrown during the workflow and treat as an invalid result by scoring 0
        # Note: scoring 0 may be problematic if the evaluator's range includes negative numbers
        print(f"Workflow execution threw error: {e}. Automatic score of 0")
    finally:
        q.put((tid, price!=0, result, score, price, None, exec_time))
        sema.release()
        
from cognify.optimizer.checkpoint import pbar_utils

def get_config_hash(config):
    config_cpy = copy.deepcopy(config)
    ms_keys = [
        'generate_query_0_model_selection',
        'generate_query_1_model_selection',
        'generate_answer_model_selection',
    ]
    for k in ms_keys:
        if config_cpy[k].startswith('None_fireworks_ai'):
            config_cpy[k] = 'None_' + old_model
    dict_str = json.dumps(config_cpy, sort_keys=True)
    hash_object = hashlib.sha256(dict_str.encode('utf-8'))
    
    # Return the hexadecimal hash
    return hash_object.hexdigest()

def try_one_config(cid, config):
    sema = mp.BoundedSemaphore(batch)
    result_q = mp.Queue()
    results = []
    all_workers = []
    
    pbar_utils.add_pbar(
        name=str(cid),
        desc=f"Eval config {cid}",
        total=len(devset),
        initial=0,
        leave=False,
        indent=0,
    )
    total_score, total_cost, total_exec_time, n_success = 0.0, 0.0, 0.0, 0
    
    def update_pbar(eval_result):
        nonlocal total_score, total_cost, total_exec_time, n_success
        n_success += 1
        score, price = eval_result[3], eval_result[4]
        total_score += score
        total_cost += price
        total_exec_time += eval_result[6]
        pbar_utils.add_opt_progress(
            name=str(cid),
            score=total_score / n_success,
            price=total_cost / n_success,
            exec_time=total_exec_time / n_success,
            total_cost=total_cost,
            is_evaluator=True,
        )
    
    for i, (input, label) in enumerate(devset):
        worker = mp.Process(
            target=run_once,
            args=(i, input, label, config, cog_space, sema, result_q),
        )
        worker.start()
        all_workers.append(worker)
        
    for i in range(len(all_workers)):
        result = result_q.get()
        if not result[1]:
            continue
        results.append(result)
        update_pbar(result)
    
    for worker in all_workers:
        worker.join()
    
    pbar_utils.close_pbar(str(cid))
        
    tids = []
    prices = []
    scores = []
    demos = []
    exec_times = []
    for tid, finished, result, score, price, demo, exec_time in results:
        assert finished, "Only finished tasks should be collected"
        tids.append(tid)
        prices.append(price)
        scores.append(score)
        demos.append(demo)
        exec_times.append(exec_time)
    reduced_score = sum(scores) / len(scores)
    reduced_price = sum(prices) / len(prices)
    reduced_exec_time = sum(exec_times) / len(exec_times)
    eval_result = EvaluationResult(
        ids=tids,
        scores=scores,
        prices=prices,
        exec_times=exec_times,
        total_eval_cost=sum(prices),
        complete=len(results) == len(devset),
        reduced_score=reduced_score,
        reduced_price=reduced_price,
        reduced_exec_time=reduced_exec_time,
    )
    with open(f"grid_results/{get_config_hash(config)}.json", "w") as f:
        output = {
            "config": config,
            "results": eval_result.to_dict(),
        }
        json.dump(output, f, indent=4)
           
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import os
import random

def grid_search():
    # manually try different combinations of reasoning, model selection and few-shot params
    # set combination for each agent
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    to_search = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        if not os.path.exists(f"grid_results/{get_config_hash(config)}.json"):
            to_search.append(config)
    to_search = random.sample(to_search, 500)
    
    futures = []
    pbar_utils.add_pbar(
        name="main bar",
        desc=f"Overall Progress: ",
        total=len(to_search),
        initial=0,
        leave=True,
        indent=0,
    )
    
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        for i, config in enumerate(to_search):
            futures.append(executor.submit(try_one_config, i, config))
        
        for future in as_completed(futures):
            pbar_utils.add_normal_progress("main bar")
    
grid_search()