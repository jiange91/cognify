#================================================================
# Evaluator
#================================================================

import cognify
from cognify.hub.evaluators import f1_score_str

@cognify.register_evaluator
def evaluate_answer(answer, label):
    return f1_score_str(answer, label)

#================================================================
# Data Loader
#================================================================

import json

@cognify.register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = {
            'question': d["question"], 
            'documents': d["docs"]
        }
        output = {
            'label': d["label"],
        }
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.optimizer.core import driver, flow
from cognify.llm.model import LMConfig
from cognify.hub.cogs import reasoning, model_selection
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter

reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
params = [reasoning_param]
outer_loop_config = driver.LayerConfig(
    layer_name="structure",
    universal_params=params,
    expected_num_agents=3,
)

model_selection_param = model_selection.model_selection_factory(
    [
        LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-e62eec4a', kwargs={'max_tokens': 1024}),
        LMConfig(model='gpt-4o-mini', kwargs={'max_tokens': 1024}),
    ]
)
middle_loop_config = driver.LayerConfig(
    layer_name="step",
    universal_params=[model_selection_param],
    expected_num_agents=3,
)

few_shot_params = LMFewShot(2)
inner_loop_config = driver.LayerConfig(
    layer_name="weight",
    universal_params=[few_shot_params, reasoning_param],
    expected_num_agents=3,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, middle_loop_config, inner_loop_config],
    opt_history_log_dir="test_setup",
    evaluator_batch_size=10,
    quality_constraint=None,
    auto_set_layer_config=True,
    total_num_trials=64,
)