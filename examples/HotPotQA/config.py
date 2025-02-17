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
    
    trainset = [formatting(x) for x in dataset.train[0:10]]
    valset = [formatting(x) for x in dataset.train[100:150]]
    devset = [formatting(x) for x in dataset.dev]
    return trainset, valset, devset

from cognify.hub.search import default
# search_settings = default.create_search(
#     n_trials=20,
#     evaluator_batch_size=50,
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
inner_opt_config = flow.OptConfig(
    n_trials=0, # does not matter, outer will set
)
params = [reasoning_param, few_shot_params]
inner_loop_config = driver.LayerConfig(
    layer_name="inner",
    universal_params=params,
    opt_config=inner_opt_config,
)

model_selection_param = model_selection.model_selection_factory(
    [
        LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-f079996b', kwargs={'max_tokens': 1024}),
        LMConfig(model='gpt-4o-mini', kwargs={'max_tokens': 1024}),
    ]
)
middle_loop_config = driver.LayerConfig(
    layer_name="middle",
    universal_params=[model_selection_param],
    opt_config=flow.OptConfig(
        n_trials=4, # does not matter, outer will set
        use_HB_allocation=True,
    ),
)

general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
general_ensemble_params = ensemble.ModuleEnsemble(
    [NoChange(), general_usc_ensemble]
)
outer_opt_config = flow.OptConfig(
    n_trials=4,
    use_SH_allocation=True,
)
outer_loop_config = driver.LayerConfig(
    layer_name="outer",
    universal_params=[general_ensemble_params],
    opt_config=outer_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, middle_loop_config, inner_loop_config],
    opt_history_log_dir="hb_with_models",
    evaluator_batch_size=50,
    quality_constraint=0.99,
)