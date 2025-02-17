import argparse
import json
from datetime import datetime
import os
import debugpy
import multiprocessing as mp
import sys
from src.utils import parse_arguments, read_from_file

import cognify
import numpy as np

@cognify.register_data_loader
def load_data():
    args = parse_arguments()
    all_train = read_from_file('data/dev/other_sub_sampled.json', args)
    test_set = read_from_file('data/dev/sub_sampled_bird_dev_set.json', args)
    
    # shuffle the data
    # all_train = np.random.permutation(all_train).tolist()
    # return all_train[:100], all_train[100:], test_set[:10]
    return test_set, None, test_set


@cognify.register_evaluator
def eval(stats):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in stats['counts'].values())
    return 1.0 if correct else 0.0

from cognify.hub.search import text_to_sql
# search_settings = text_to_sql.create_search(opt_log_dir="cognify_opt_debit_card", evaluator_batch_size=30)

# from cognify.hub.search import default
# search_settings = default.create_search(
#     search_type='light',
#     n_trials=15,
#     opt_log_dir='ca_school_opt_demo',
#     evaluator_batch_size=40,
# )

from cognify.optimizer.core import driver, flow
from cognify.llm.model import LMConfig
from cognify.hub.cogs import reasoning, ensemble, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter

reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
few_shot_params = LMFewShot(4)
model_selection_param = model_selection.model_selection_factory(
    [
        LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-4917141b'),
        LMConfig(model='gpt-4o-mini'),
    ]
)
inner_opt_config = flow.OptConfig(
    n_trials=16, # does not matter, outer will set
    throughput=2,
)
params = [reasoning_param, few_shot_params, model_selection_param]
inner_loop_config = driver.LayerConfig(
    layer_name="inner",
    universal_params=params,
    opt_config=inner_opt_config,
)

def add_ensemble_option(lm_name):
    usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    ensemble_param = ensemble.ModuleEnsemble(
        options=[NoChange(), usc_ensemble]
    )
    ensemble_param.module_name = lm_name
    return ensemble_param

ensemble_params = [
    add_ensemble_option('table_selection'),
    add_ensemble_option('candidate_generation'),
    add_ensemble_option('revision'),
]

outer_opt_config = flow.OptConfig(
    n_trials=4,
    throughput=2,
)
outer_loop_config = driver.LayerConfig(
    layer_name="outer",
    dedicate_params=ensemble_params,
    opt_config=outer_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, inner_loop_config],
    opt_history_log_dir="b64_l2_4_14",
    evaluator_batch_size=10,
    quality_constraint=1.00,
    eval_time_out=180,
)