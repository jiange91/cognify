import optuna
import copy
import os

from cognify.optimizer.core import driver, flow
from cognify.llm.model import LMConfig
from cognify.hub.cogs import reasoning, model_selection, ensemble
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter
from cognify.hub.cogs.common import CogBase

from cognify.optimizer.core.opt_layer import OptLayer, GlobalOptConfig
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_exec_time_reduction

GlobalOptConfig.base_quality = 0.22
GlobalOptConfig.base_price = 0.36
GlobalOptConfig.base_exec_time = 0.4

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
    [NoChange(), general_usc_ensemble]
)
non_structure_cogs: list[CogBase] = [reasoning_param, model_selection_param, few_shot_param]

agent_names = ["generate_query_0", "generate_query_1", "generate_answer"]

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
from config import load_data_minor

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

opt_log_dir = 'one_layer_test'
# create the log dir if not exists
if not os.path.exists(opt_log_dir):
    os.makedirs(opt_log_dir, exist_ok=True)

def add_constraint(trial, score):
    constraint_result = (GlobalOptConfig.base_quality - score,)
    trial.set_user_attr("constraint_result", constraint_result)

def get_constraint_result(trial):
    return trial.user_attrs.get("constraint_result", (1,))

from optuna.samplers import TPESampler
from cognify.optimizer.checkpoint.ckpt import TrialLog, LogManager
_log_mng = LogManager(opt_log_dir)

class OneLayer():
    def __init__(self):
        sampler = TPESampler(
            multivariate=True,
            n_startup_trials=5,
            constraints_func=get_constraint_result,
        )
        self.study = optuna.create_study(sampler=sampler, directions=optimize_directions)
        self.n_trials = 3
        self.script_path = "workflow.py"
        self._id = "one_layer"
        LogManager().register_layer(
            self._id, self._id,
            f"{opt_log_dir}/opt_log.json", 
            True
        )
        LogManager().layer_stats[self._id].init_progress_bar(
            level=0,
            budget=self.n_trials,
            leave=True,
        )
    
    def objective(self, trial):
        structure_decisions: dict[str, list[tuple[str, str]]] = {}
        other_decisions: dict[str, list[tuple[str, str]]] = {}
        all_decisions = {}
        for sc_name in structure_cogs:
            agent_name, options = structure_cogs[sc_name]
            sv = trial.suggest_categorical(sc_name, options)
            structure_decisions[agent_name] = [(all_cogs[sc_name].name, sv)]
            all_decisions[sc_name] = sv
            for oc_name in other_cog_table[sc_name][sv]:
                agent_name, options = other_cog_table[sc_name][sv][oc_name]
                ov = trial.suggest_categorical(oc_name, options)
                all_decisions[oc_name] = ov
                if agent_name not in other_decisions:
                    other_decisions[agent_name] = [(all_cogs[oc_name].name, ov)]
                else:
                    other_decisions[agent_name].append((all_cogs[oc_name].name, ov))
        # print(structure_decisions)
        # print(other_decisions)
        
        eval_result = self.eval_workflow(structure_decisions, other_decisions, all_decisions)
        add_constraint(trial, eval_result.reduced_score)
        return eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time

    def eval_workflow(self, structure_decisions, other_decisions, all_decisions):
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
        log_id = LogManager().add_trial(self._id, all_decisions)
        LogManager().layer_stats[self._id].opt_logs[log_id].eval_task_dict = eval_task.to_dict()
        eval_result = evaluator.get_score(
            mode='train',
            task=eval_task,
            show_process=True,
        )
        LogManager().report_trial_result(self._id, log_id, eval_result)
        LogManager().layer_stats[self._id].save_opt_logs()
        return eval_result
        
    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials)
        LogManager()._save_opt_trace()
        opt_cost, pareto_frontier, finished_opt_logs = LogManager().get_global_summary(verbose=True)
        self.dump_frontier_details(pareto_frontier, finished_opt_logs)
        print(f"Optimal cost: {opt_cost}")
        
    def dump_frontier_details(self, frontier, finished_opt_logs):
        param_log_dir = os.path.join(opt_log_dir, "pareto_frontier_details")
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir, exist_ok=True)
        for i, trial_log in enumerate(frontier):
            trial_log: TrialLog
            score, price, exec_time = trial_log.result.reduced_score, trial_log.result.reduced_price, trial_log.result.reduced_exec_time
            dump_path = os.path.join(param_log_dir, f"Pareto_{i+1}.cog")
            details = f"Trial - {trial_log.id}\n"
            log_path = finished_opt_logs[trial_log.id][1]
            details += f"Log at: {log_path}\n"
            if GlobalOptConfig.base_quality is not None:
                details += _report_quality_impv(score, GlobalOptConfig.base_quality)
            if GlobalOptConfig.base_price is not None:
                details += _report_cost_reduction(price, GlobalOptConfig.base_price)
            if GlobalOptConfig.base_exec_time is not None:
                details += _report_exec_time_reduction(exec_time, GlobalOptConfig.base_exec_time)
                
            details += f"Quality: {score:.3f}, Cost per 1K invocation: ${price * 1000:.2f}, Avg exec time: {exec_time:.2f} s\n"
            trans = trial_log.show_transformation()
            details += trans
            with open(dump_path, "w") as f:
                f.write(details)
        
OneLayer().optimize()