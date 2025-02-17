import optuna
import copy

from cognify.optimizer.core import driver, flow
from cognify.llm.model import LMConfig
from cognify.hub.cogs import reasoning, model_selection, ensemble
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter
from cognify.hub.cogs.common import CogBase

reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
model_selection_param = model_selection.model_selection_factory(
    [
        # LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-33abb831', kwargs={'max_tokens': 1024}),
        LMConfig(model='gpt-4o-mini', kwargs={'max_tokens': 1024}),
    ]
)
few_shot_param = LMFewShot(2)
general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
general_ensemble_param = ensemble.ModuleEnsemble(
    [NoChange()]
)
non_structure_cogs: list[CogBase] = [reasoning_param, model_selection_param, few_shot_param]

agent_names = ["qa_agent"]

all_cogs: dict[str, CogBase] = {}
structure_cogs = {}
other_cog_table = {}

def spwan_agent(agent_name, pool):
    for param in non_structure_cogs:
        nd_param = copy.deepcopy(param)
        nd_param.module_name = agent_name
        pool[nd_param.hash] = (agent_name, list(nd_param.options.keys()))
        all_cogs[nd_param.hash] = nd_param

def build_search_space():
    for module_name in agent_names:
        d_param = copy.deepcopy(general_ensemble_param)
        d_param.module_name = module_name
        structure_cogs[d_param.hash] = (module_name, list(d_param.options.keys()))
        all_cogs[d_param.hash] = d_param
        
        other_cog_table[d_param.hash] = {'NoChange': {}, 'universal_self_consistency': {}}
        spwan_agent(module_name, other_cog_table[d_param.hash]['NoChange'])
        
        for i in range(3):
            spwan_agent(f"{module_name}_sampler_{i}", other_cog_table[d_param.hash]['universal_self_consistency'])
        spwan_agent(f"{module_name}_aggregator", other_cog_table[d_param.hash]['universal_self_consistency'])
        
build_search_space()
# print(structure_cogs)
# print(other_cog_table)
from config import load_data_minor, evaluate_answer

train_set, val_set, test_set = load_data_minor()

from cognify.optimizer.evaluator import EvaluatorPlugin, EvalTask

evaluator = EvaluatorPlugin(
    trainset=train_set,
    evalset=val_set,
    testset=None,
    evaluator_path="config.py",
    n_parallel=10,
)

optimize_directions = [
    "maximize", # quality
    "minimize", # cost
    "minimize", # exec time
]

from optuna.samplers import TPESampler
from cognify.optimizer.checkpoint.ckpt import TrialLog, LogManager

class OneLayer():
    def __init__(self):
        sampler = TPESampler(
            multivariate=True,
            n_startup_trials=5,
            # constraints_func=constraint,
        )
        self.study = optuna.create_study(sampler=sampler, directions=optimize_directions)
        self.n_trials = 10
        self.script_path = "workflow.py"
        LogManager().register_layer(self.name, self._id, self.top_down_info.opt_config.opt_log_path, self.is_leaf)
    
    def objective(self, trial):
        structure_decisions: dict[str, list[tuple[str, str]]] = {}
        other_decisions: dict[str, list[tuple[str, str]]] = {}
        for sc_name in structure_cogs:
            agent_name, options = structure_cogs[sc_name]
            sv = trial.suggest_categorical(sc_name, options)
            structure_decisions[agent_name] = [(all_cogs[sc_name].name, sv)]
            for oc_name in other_cog_table[sc_name][sv]:
                agent_name, options = other_cog_table[sc_name][sv][oc_name]
                ov = trial.suggest_categorical(oc_name, options)
                if agent_name not in other_decisions:
                    other_decisions[agent_name] = [(all_cogs[oc_name].name, ov)]
                else:
                    other_decisions[agent_name].append((all_cogs[oc_name].name, ov))
        print(structure_decisions)
        print(other_decisions)
        
        return self.eval_workflow(structure_decisions, other_decisions)

    def eval_workflow(self, structure_decisions, other_decisions):
        eval_task = EvalTask(
            script_path=self.script_path,
            args=[],
            other_python_paths=[],
            all_params=copy.deepcopy(all_cogs),
            module_name_paths=None,
            aggregated_proposals={
                'structure': structure_decisions,
                'other': other_decisions,
            }
        )
        eval_result = evaluator.get_score(
            mode='train',
            task=eval_task,
            show_process=True,
        )
        return eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time
        
    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials)
        print(self.study.best_trials)
        
OneLayer().optimize()