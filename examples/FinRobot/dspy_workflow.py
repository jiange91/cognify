import os
import dotenv
dotenv.load_dotenv()

import dspy

gpt4o_mini = dspy.LM(model='gpt-4o-mini', max_tokens=1024)

dspy.configure(lm=gpt4o_mini)

def format_history(history: list[str]):
    if not history:
        return "Empty"
    hist_str = ""
    for i, hist in enumerate(history):
        hist_str += f"\n---- Step {i+1} ----\n{hist}"
    # print(f"formatted history: {hist_str}")
    return hist_str

from agent_pool_dspy import agents
from collections import defaultdict

class FinAgent(dspy.Module):
    def __init__(self, agent_name):
        super().__init__()
        
        self.fin_robot = agents[agent_name]
        self.task_history = defaultdict(list)
    
    def forward(self, order, task):
        hist = format_history(self.task_history[task])
        response = self.fin_robot(history=hist, current_order=order).response
        self.task_history[task].append(
            f"Order: {order}\n"
            f"My Response: {response}"
        )
        return response
 
import json   
with open('agent_profiles.json') as f:
    profiles = json.load(f)
    
group_members: dict[str, FinAgent] = {}
for profile in profiles:
    agent_name = profile['name']
    group_members[agent_name] = FinAgent(agent_name.lower())

from leader_dspy import leader_agent, LeaderResponse
import re

def parse_order_string(order_string: str):
    pattern = r"\[(.*?)\]\s+(.*)"
    match = re.search(pattern, order_string)
    
    if match:
        name = match.group(1)  # Extract name inside square brackets
        order = match.group(2)  # Extract the order instruction
        return name, order
    else:
        raise ValueError("Invalid order string format. Ensure it follows '[<name of staff>] <order>'.")

class FinRobot(dspy.Module):
    def __init__(self, leader_agent, group_members, k=5):
        super().__init__()
        self.leader_agent = leader_agent
        self.group_members = group_members
        self.k = k
        self.task_history = defaultdict(list)
    
    def forward(self, task):
        # print(task)
        for i in range(self.k):
            # Leader assigns a task to a group member
            project_hist = format_history(self.task_history[task])
            leader_msg: LeaderResponse = self.leader_agent(
                task=task, project_history=project_hist, remaining_order_budget=self.k - i
            ).response
            # print(leader_msg)
            if leader_msg.project_status == "END":
                return leader_msg.solution
            
            member_name, member_order = parse_order_string(leader_msg.member_order)
            member_response = self.group_members[member_name](order=member_order, task=task)
            # print(f"member response: {member_response}")
            self.task_history[task].append(
                # f"Status: {leader_msg.project_status}\n"
                f"Order: {member_name} - {member_order}\n"
                f"Member Response: {member_response}"
            )
        else:
            return "Project not completed in time."
        
fin_robot = FinRobot(leader_agent=leader_agent, group_members=group_members, k=3)

if __name__ == "__main__":
    task = """
    Goal: Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.
    
    phrase: NEW YORK (Reuters) - Apple Inc Chief Executive Steve Jobs sought to soothe investor concerns about his health on Monday, saying his weight loss was caused by a hormone imbalance that is relatively simple to treat.
    """
    print(fin_robot(task=task))