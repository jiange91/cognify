import cognify
from cognify.hub.evaluators import f1_score_str

@cognify.register_evaluator
def answer_f1(answer: str, ground_truth: str):
    return f1_score_str(answer, ground_truth)

def formatting(item):
    return (
        {'question': item.question},
        {'ground_truth': item.answer}
    )

@cognify.register_data_loader
def load_data_minor():
    from dspy.datasets.hotpotqa import HotPotQA
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    
    trainset = [formatting(x) for x in dataset.train[0:100]]
    valset = [formatting(x) for x in dataset.train[100:150]]
    devset = [formatting(x) for x in dataset.dev]
    return trainset, valset, devset

from cognify.hub.search import default
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
        LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-b9919f71', kwargs={'max_tokens': 1024}),
        LMConfig(model='gpt-4o-mini', kwargs={'max_tokens': 1024}),
    ]
)
params = [reasoning_param, few_shot_params, model_selection_param]

inner_opt_config = flow.OptConfig(
    n_trials=16, # does not matter, outer will set
)
inner_loop_config = driver.LayerConfig(
    layer_name="inner",
    universal_params=params,
    opt_config=inner_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[inner_loop_config],
    opt_history_log_dir="light_16",
    evaluator_batch_size=50,
    quality_constraint=1.0,
)