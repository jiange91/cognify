import re
from collections import defaultdict
from prompts import leader_system_message, role_system_message
import json
from pydantic import BaseModel
import dotenv
import cognify

dotenv.load_dotenv()

lm_selection_ori = 'gpt-4o'
lm_config = cognify.LMConfig(model=lm_selection_ori, kwargs={"temperature": 0.0})

def format_history(history: list[str]):
    if not history:
        return "Empty"
    hist_str = ""
    for i, hist in enumerate(history):
        hist_str += f"\n---- Step {i+1} ----\n{hist}"
    # print(f"formatted history: {hist_str}")
    return hist_str
    
class FinAgent:
    def __init__(self, agent_name, agent_role):
        self.fin_robot = cognify.Model(
            agent_name=agent_name,
            system_prompt=role_system_message.format(name=agent_name, responsibilities=agent_role),
            input_variables=[cognify.Input(name="history"), cognify.Input(name="current_order")],
            output=cognify.OutputLabel(name="response"),
            lm_config=lm_config
        )
        self.task_history = defaultdict(list)
    
    def solve_order(self, order, task):
        hist = format_history(self.task_history[task])
        response = self.fin_robot(
            inputs={"history": hist, "current_order": order}
        )
        self.task_history[task].append(
            f"Order: {order}\n"
            f"My Response: {response}"
        )
        return response

# Load group members
with open('agent_profiles.json') as f:
    profiles = json.load(f)
group_members: dict[str, FinAgent] = {}
for profile in profiles:
    agent_name = profile['name']
    agent_role = profile['profile']
    group_members[agent_name] = FinAgent(agent_name, agent_role)
    
# Set up group leader
class LeaderResponse(BaseModel):
    project_status: str
    member_order: str
    solution: str

group_desc = "\n".join(["- {name}: {profile}".format(**member) for member in profiles])
group_leader = cognify.StructuredModel(
    agent_name="group_leader",
    system_prompt=leader_system_message.format(group_desc=group_desc),
    input_variables=[
        cognify.Input(name="task"), 
        cognify.Input(name="project_history"), 
        cognify.Input(name="remaining_order_budget")
    ],
    output_format=cognify.OutputFormat(schema=LeaderResponse),
    lm_config=lm_config,
)

def parse_order_string(order_string: str):
    pattern = r"\[(.*?)\]\s+(.*)"
    match = re.search(pattern, order_string)
    
    if match:
        name = match.group(1)  # Extract name inside square brackets
        order = match.group(2)  # Extract the order instruction
        return name, order
    else:
        raise ValueError("Invalid order string format. Ensure it follows '[<name of staff>] <order>'.")
    
class FinRobot:
    def __init__(self, k=5):
        self.k = k
        self.task_history = defaultdict(list)
    
    def solve_task(self, task):
        # print(task)
        for i in range(self.k):
            # Leader assigns a task to a group member
            project_hist = format_history(self.task_history[task])
            leader_msg: LeaderResponse = group_leader(
                inputs={"task": task, "project_history": project_hist, "remaining_order_budget": self.k - i}
            )
            # print(leader_msg)
            if leader_msg.project_status == "END":
                return leader_msg.solution
            
            member_name, member_order = parse_order_string(leader_msg.member_order)
            member_response = group_members[member_name].solve_order(member_order, task)
            # print(f"member response: {member_response}")
            self.task_history[task].append(
                # f"Status: {leader_msg.project_status}\n"
                f"Order: {member_name} - {member_order}\n"
                f"Member Response: {member_response}"
            )
        else:
            return "Project not completed in time."

fin_robot = FinRobot(k=3)

@cognify.register_workflow
def solve_fin_task(task: str, mode):
    # Solve task iteratively
    # In each round, the leader assigns a task to a group member or decide to complete the task
    answer = fin_robot.solve_task(task)
    return {"answer": answer}

if __name__ == "__main__":

    solve_fin_task(
        # """
        # What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.
        
        # "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano",
        # """
        
        # """
        # What is considered a business expense on a business trip?
        # """
        
        """
        Goal: Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.
        
        phrase: NEW YORK (Reuters) - Apple Inc Chief Executive Steve Jobs sought to soothe investor concerns about his health on Monday, saying his weight loss was caused by a hormone imbalance that is relatively simple to treat.
        """,
        None
    )