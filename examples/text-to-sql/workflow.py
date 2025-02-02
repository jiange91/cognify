# ⚠️ USE AT YOUR OWN RISK
# first: pip install pysqlite3-binary
# then in settings.py:

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from datetime import datetime
from src.pipeline.workflow_builder import (
    build_pipeline,
    keyword_extraction,
    entity_retrieval,
    context_retrieval,
    column_filtering,
    table_selection,
    column_selection,
    candidate_generation,
    revision,
    evaluation
)

from src.runner.run_manager import RunManager
from src.runner.task import Task
from src.utils import parse_arguments
import cognify

@cognify.register_workflow
def worker_opt(args, dataset):
    """
    Main function to run the pipeline with the specified configuration.
    """
    assert len(dataset) == 1, "Worker process perform one task at a time"
    
    run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task = Task(dataset[0])
    result_dir = f"light_4_result/{task.db_id}/{task.question_id}/{run_start_time}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    run_manager = RunManager(args, result_dir)
    run_manager.initialize_tasks(dataset)
    task = run_manager.tasks[0]
    
    result = run_manager.worker(task)
    run_manager.task_done(result, show_progress=False) 

    return {'stats': run_manager.statistics_manager.statistics.to_dict()}

# @cognify.register_workflow
# def worker_demo(query):
#     """
#     Main function to run the pipeline with the specified configuration.
#     """
#     args = parse_arguments()
#     run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     task = Task(query)
#     result_dir = f"demo_one/{task.db_id}/{task.question_id}/{run_start_time}"
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir, exist_ok=True)

#     run_manager = RunManager(args, result_dir)
#     run_manager.initialize_tasks([query])
#     task = run_manager.tasks[0]
    
#     result = run_manager.worker(task)
#     run_manager.task_done(result, show_progress=False) 

#     return {'stats': run_manager.statistics_manager.statistics.to_dict()}

