import os
import json
from typing import Union, Optional, Callable
import logging
import re
import math
import copy

from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
)
from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core.flow import TopDownInformation, LayerConfig
from cognify.optimizer.core.opt_layer import OptLayer, GlobalOptConfig
from cognify.optimizer.checkpoint.ckpt import LogManager, TrialLog
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_exec_time_reduction

logger = logging.getLogger(__name__)

def get_layer_evaluator_factory(
    next_layer_factory, 
    layer_config: LayerConfig,
    level: int,
    is_leaf: bool
):
    def _factory():
        return OptLayer(
            name=layer_config.layer_name,
            opt_config=layer_config.opt_config,
            hierarchy_level=level,
            is_leaf=is_leaf,
            next_layer_factory=next_layer_factory,
            dedicate_params=layer_config.dedicate_params,
            universal_params=layer_config.universal_params,
            target_modules=layer_config.target_modules,
        )
    return _factory

class MultiLayerOptimizationDriver:
    def __init__(
        self,
        control_param: ControlParameter,
        base_result: EvaluationResult,
    ):
        """Driver for multi-layer optimization

        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        GlobalOptConfig.eval_time_out = control_param.eval_time_out
        if control_param.quality_constraint is None:
            GlobalOptConfig.quality_constraint = None
        else:
            GlobalOptConfig.quality_constraint = control_param.quality_constraint * base_result.reduced_score
        
        if base_result is not None:
            GlobalOptConfig.base_quality = base_result.reduced_score
            GlobalOptConfig.base_price = base_result.reduced_price
            GlobalOptConfig.base_exec_time = base_result.reduced_exec_time
        _log_mng = LogManager(control_param.opt_history_log_dir)

        # initialize optimization layers
        self._set_layer_config(control_param)
        with open(os.path.join(control_param.opt_history_log_dir, 'actual_search_layer_config.json'), 'w') as f:
            configs = [layer_config.to_dict() for layer_config in self.layer_configs]
            json.dump(configs, f, indent=4)
        self.opt_layer_factories: list[Callable] = [None] * (len(self.layer_configs) + 1)

        self.opt_log_dir = control_param.opt_history_log_dir

        # config log dir for layer opts
        # NOTE: only the top layer will be set, others are decided at runtime
        self.layer_configs[0].opt_config.log_dir = os.path.join(
            self.opt_log_dir, self.layer_configs[0].layer_name
        )
        # NOTE: since these will be set at runtime, we set them to None
        for layer_config in self.layer_configs[1:]:
            layer_config.opt_config.log_dir = None
            layer_config.opt_config.opt_log_path = None
            layer_config.opt_config.param_save_path = None
        
    def _set_layer_config(self, control_param: ControlParameter):
        opt_layer_configs = control_param.opt_layer_configs
        if not control_param.auto_set_layer_config:
            self.layer_configs = opt_layer_configs
        else:
            # decide search space partition
            # TODO: consider change interface to accept entire cog space instead of layer definition
            assert len(opt_layer_configs) == 3, "Always define structure, step, weight layers"
            assert control_param.total_num_trials is not None, "Please inform the total number of trials"
            
            # get Ei
            expected_trials = []
            dimensions = []
            option_sizes = []
            for i, layer_config in enumerate(opt_layer_configs):
                if layer_config is None:
                    dimensions.append(0)
                    option_sizes.append(0)
                    expected_trials.append(1)
                    continue
                if layer_config.universal_params:
                    if layer_config.target_modules is None and layer_config.expected_num_agents is None:
                        raise ValueError(
                            "If you want to use search space auto-partition while using universal params, "
                            "please inform the number of agents or set target module names"
                        )
                    num_agent = layer_config.expected_num_agents or len(layer_config.target_modules)
                    universal_cog_space = num_agent * len(layer_config.universal_params)
                    universal_option_size = num_agent * sum(len(param.options) for param in layer_config.universal_params)
                else:
                    universal_cog_space = 0
                    universal_option_size = 0
                dimensions.append(universal_cog_space + len(layer_config.dedicate_params))
                dedicate_option_size = sum(len(param.options) for param in layer_config.dedicate_params)
                option_sizes.append(universal_option_size + dedicate_option_size)
                ei = math.ceil(max(dimensions[-1] ** 1.2, option_sizes[-1]))
                expected_trials.append(max(ei, 1))
            logger.debug(f"Expected budgets before partition: {expected_trials}")
            
            # search space partition
            three_layer_budgets = math.prod(expected_trials)
            if control_param.total_num_trials < three_layer_budgets:
                # merge step and weight layer
                step_layer = opt_layer_configs[1]
                weight_layer = opt_layer_configs[2]
                
                # to avoid target-module conflict, transform all params to valid dedicate params
                def _clean_params(layer: LayerConfig):
                    if layer is None:
                        return [], []
                    if layer.target_modules is None:
                        return layer.dedicate_params, layer.universal_params
                    dedicate_params = [cog for cog in layer.dedicate_params if cog.module_name in layer.target_modules]
                    for cog in layer.universal_params:
                        for module_name in layer.target_modules:
                            dcog = copy.deepcopy(cog)
                            dcog.module_name = module_name
                            dedicate_params.append(dcog)
                    return dedicate_params, []
                    
                s_d, s_u = _clean_params(step_layer)
                w_d, w_u = _clean_params(weight_layer)
                if not step_layer or not weight_layer:
                    layer_name = step_layer.layer_name if step_layer else weight_layer.layer_name
                else:
                    layer_name = f"{step_layer.layer_name}_and_{weight_layer.layer_name}"
                inner_layer = LayerConfig(
                    layer_name=layer_name,
                    dedicate_params=s_d + w_d,
                    universal_params=s_u + w_u,
                )
                opt_layer_configs = [opt_layer_configs[0], inner_layer]
                dimensions = [dimensions[0], dimensions[1] + dimensions[2]]
                option_sizes = [option_sizes[0], option_sizes[1] + option_sizes[2]]
                expected_trials = [max(math.ceil(d ** 1.2), o, 1) for d, o in zip(dimensions, option_sizes)]
            
            # filter out None layer
            opt_layer_configs = [layer for layer, dim in zip(opt_layer_configs, dimensions) if dim > 0]
            expected_trials = [exp for exp, dim in zip(expected_trials, dimensions) if dim > 0]
            logger.debug(f"Expected budgets after partition: {expected_trials}")
            # search budget partition
            B_prime = math.prod(expected_trials)
            allocations = [0] * len(opt_layer_configs) + [1]
            if control_param.total_num_trials >= B_prime:
                # give each layer budget proportial to expected trials
                for i, (exp, layer_config) in enumerate(zip(expected_trials, opt_layer_configs)):
                    layer_config.opt_config.n_trials = int(exp * (control_param.total_num_trials / B_prime) ** (1 / len(opt_layer_configs)))
                    allocations[i] = layer_config.opt_config.n_trials
            else:
                # give lower layer greedy budget
                for i in range(len(opt_layer_configs) - 1, -1, -1):
                    opt_layer_configs[i].opt_config.n_trials = min(
                        expected_trials[i],
                        int(control_param.total_num_trials / math.prod(allocations[i+1:]))
                    )
                    allocations[i] = opt_layer_configs[i].opt_config.n_trials

            # set R for each layer
            for i, layer_config in enumerate(opt_layer_configs):
                # set R for each layer
                layer_config.opt_config.initial_step_budget = math.ceil(
                    allocations[i+1] / layer_config.opt_config.prune_rate
                )
            self.layer_configs = opt_layer_configs
            
            
    def build_tiered_optimization(self, evaluator: EvaluatorPlugin):
        """Build tiered optimization from bottom to top"""
        self.opt_layer_factories[-1] = lambda: evaluator
        
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - ri - 1
            next_layer_factory = self.opt_layer_factories[idx + 1]
            current_layer_factory = get_layer_evaluator_factory(next_layer_factory, layer_config, idx, ri == 0)
            self.opt_layer_factories[idx] = current_layer_factory

    def run(
        self,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[tuple[TrialLog, str]], dict[str, TrialLog]]:
        self.build_tiered_optimization(evaluator)
        top_layer = self.opt_layer_factories[0]()
        logger.info("----------------- Start Optimization -----------------")
        top_layer.easy_optimize(
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        logger.info("----------------- Optimization Finished -----------------")
        LogManager()._save_opt_trace()
        return self.inspect(dump_details=True)

    def _extract_trial_id(self, config_id: str) -> str:
        param_log_dir = os.path.join(self.opt_log_dir, "pareto_frontier_details")
        if not os.path.exists(param_log_dir):
            raise ValueError(
                f"Cannot find the optimization log directory at {param_log_dir}"
            )

        with open(os.path.join(param_log_dir, f"{config_id}.cog"), "r") as f:
            first_line = f.readline().strip()
        match = re.search(r"Trial - (.+)", first_line)
        if match:
            trial_id = match.group(1)
            return trial_id
        else:
            raise ValueError(
                f"Cannot extract trial id from the log file {config_id}.cog"
            )

    def _find_config_log_path(self, trial_id: str) -> str:
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        tdi = TopDownInformation(
            opt_config=opt_config,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=None,
            script_args=None,
            other_python_paths=None,
        )

        top_layer = self.opt_layers[0]
        top_layer.load_opt_log(opt_config.opt_log_path)
        top_layer.top_down_info = tdi
        all_configs = top_layer.get_all_candidates()
        config_path = None

        for opt_log, path in all_configs:
            if opt_log.id == trial_id:
                config_path = path
                break
        else:
            raise ValueError(f"Config {trial_id} not found in the optimization log.")
        return config_path

    def evaluate(
        self,
        evaluator: EvaluatorPlugin,
        config_id: str,
    ) -> EvaluationResult:
        self.load_from_file()
        trial_id = self._extract_trial_id(config_id)
        log = LogManager().get_log_by_id(trial_id)

        # apply selected trial
        print(f"----- Testing {config_id} -----")
        # print("  Training Quality: {:.3f}, Cost per 1K invocation: ${:.2f}\n".format(trial_log.score, trial_log.price * 1000))
        
        eval_task = EvalTask.from_dict(log.eval_task_dict)
        # run evaluation
        eval_result = evaluator.get_score(mode='test', task=eval_task, show_process=True, keep_bar=True)
        
        print(f"=========== Evaluation Results ===========") 
        if GlobalOptConfig.base_quality is not None:
            print(_report_quality_impv(eval_result.reduced_score, GlobalOptConfig.base_quality))
        if GlobalOptConfig.base_price is not None:
            print(_report_cost_reduction(eval_result.reduced_price, GlobalOptConfig.base_price))
        if GlobalOptConfig.base_exec_time is not None:
            print(_report_exec_time_reduction(eval_result.reduced_exec_time, GlobalOptConfig.base_exec_time))
        print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Avg exec time: {:.2f} s".format(eval_result.reduced_score, eval_result.reduced_price * 1000, eval_result.reduced_exec_time))
        print("===========================================")

        return eval_result

    def load(
        self,
        config_id: str,
    ):
        self.load_from_file()
        trial_id = self._extract_trial_id(config_id)
        log = LogManager().get_log_by_id(trial_id)
        eval_task = EvalTask.from_dict(log.eval_task_dict)
        schema, old_name_2_new_module = eval_task.load_and_transform()
        return schema, old_name_2_new_module

    def inspect(self, dump_details: bool = False):
        self.load_from_file()
        # dump frontier details to file
        opt_cost, pareto_frontier, finished_opt_logs = LogManager().get_global_summary(verbose=True)
        if dump_details:
            self.dump_frontier_details(pareto_frontier, finished_opt_logs)
        return opt_cost, pareto_frontier, finished_opt_logs

    def dump_frontier_details(self, frontier, finished_opt_logs):
        param_log_dir = os.path.join(self.opt_log_dir, "pareto_frontier_details")
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
    
    def load_from_file(self):
        root_log = self.layer_configs[0].opt_config
        root_log.finalize()
        _log_dir_stack = [root_log.log_dir]
        leaf_layer_name = self.layer_configs[-1].layer_name
        
        while _log_dir_stack:
            log_dir = _log_dir_stack.pop()
            opt_log_path = os.path.join(log_dir, "opt_logs.json")
            
            if not os.path.exists(opt_log_path):
                continue
            with open(opt_log_path, "r") as f:
                opt_trace = json.load(f)
                
                for log_id, log in opt_trace.items():
                    layer_instance = log_id.rsplit("_", 1)[0]
                    layer_name = log["layer_name"]
                    trial_number = log_id.rsplit("_", 1)[-1]
                    sub_layer_log_dir = os.path.join(log_dir, f"{layer_name}_trial_{trial_number}")
                    _log_dir_stack.append(sub_layer_log_dir)
                    
                LogManager().register_layer(
                    layer_name=layer_name,
                    layer_instance=layer_instance,
                    opt_log_path=opt_log_path,
                    is_leaf=layer_name == leaf_layer_name,
                )
                LogManager().load_existing_logs(layer_instance)
            