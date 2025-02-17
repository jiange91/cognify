import pandas as pd
import cognify
import dotenv
from cognify.hub.evaluators import f1_score_str

dotenv.load_dotenv()

data_point = 2

def load_sentiment_data(pool):
    sentiment_df = pd.read_parquet("data/sentiment.parquet")
    data = []
    for i, row in sentiment_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'sentiment_analysis'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= data_point - 1:
            break
    pool[0].extend(data)
    pool[2].extend(data)

def load_relation_extraction_data(pool):
    relation_df = pd.read_parquet("data/relation.parquet")
    data = []
    for i, row in relation_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'relation_extraction'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= data_point - 1:
            break
    pool[0].extend(data)
    pool[2].extend(data)
    
def load_headline_data(pool):
    headline_df = pd.read_parquet("data/headline.parquet")
    data = []
    for i, row in headline_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'headline_classification'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= data_point - 1:
            break
    pool[0].extend(data)
    pool[2].extend(data)
    
def load_fiqa_data(pool):
    fiqa_df = pd.read_parquet("data/fiqa.parquet")
    data = []
    for i, row in fiqa_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'fiqa'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= data_point - 1:
            break
    pool[0].extend(data)
    pool[2].extend(data)


@cognify.register_data_loader
def load_data():
    all_train, all_val, all_test = [], None, []
    load_sentiment_data([all_train, all_val, all_test])
    # load_relation_extraction_data([all_train, all_val, all_test])
    load_headline_data([all_train, all_val, all_test])
    load_fiqa_data([all_train, all_val, all_test])
    
    return all_train, all_val, all_test


@cognify.register_evaluator
def evaluate(answer, label, mode, task):
    if mode == 'sentiment_analysis':
        return evaluate_sentiment(answer, label)
    elif mode == 'relation_extraction':
        return evaluate_relation_extraction(answer, label)
    elif mode == 'headline_classification':
        return evaluate_headline(answer, label)
    elif mode == 'fiqa':
        return evaluate_fiqa(answer, label, task)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def evaluate_sentiment(answer, label):
    return f1_score_str(answer, label)

def evaluate_relation_extraction(answer, label):
    return f1_score_str(answer, label)

def evaluate_headline(answer, label):
    return f1_score_str(answer, label)

from pydantic import BaseModel
class Assessment(BaseModel):
    success: bool

lm_config = cognify.LMConfig(model='gpt-4o-mini', kwargs={"temperature": 0.0})
qa_eval_agent = cognify.StructuredModel(
    agent_name="qa_evaluator",
    system_prompt="Given the question and the ground truth, evaluate if the response answers the question.",
    input_variables=[
        cognify.Input(name="question"), 
        cognify.Input(name="ground_truth"), 
        cognify.Input(name="response")
    ],
    output_format=cognify.OutputFormat(schema=Assessment),
    lm_config=lm_config,
)
def evaluate_fiqa(answer, label, task):
    assessment = qa_eval_agent(
        inputs={"question": task, "ground_truth": label, "response": answer}
    )
    return int(assessment.success)
    

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
        LMConfig(model='fireworks_ai/accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-33abb831', kwargs={'max_tokens': 1024}),
        LMConfig(model='gpt-4o-mini', kwargs={'max_tokens': 1024}),
    ]
)
params = [reasoning_param, few_shot_params, model_selection_param]
inner_opt_config = flow.OptConfig(
    n_trials=32,
)
inner_loop_config = driver.LayerConfig(
    layer_name="weight",
    universal_params=params,
    expected_num_agents=3,
    opt_config=inner_opt_config,
)

general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
general_ensemble_params = ensemble.ModuleEnsemble(
    [NoChange(), general_usc_ensemble]
)
outer_opt_config = flow.OptConfig(
    n_trials=4,
)
outer_loop_config = driver.LayerConfig(
    layer_name="structure",
    universal_params=[general_ensemble_params],
    expected_num_agents=3,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[inner_loop_config],
    opt_history_log_dir=f"input_sense_2",
    evaluator_batch_size=20,
    quality_constraint=1.0,
    # auto_set_layer_config=True,
    # total_num_trials=4,
)