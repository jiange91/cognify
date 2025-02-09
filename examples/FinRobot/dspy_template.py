import json
from prompts import role_system_message

with open('agent_profiles.json') as f:
    profiles = json.load(f)

def build_dspy_signature(name, profile):
    # remove underscores from name
    class_name = ''.join(word.capitalize() for word in name.replace('_', ' ').split())

    # Define the class template as a string
    class_template = f"""
class {class_name}(dspy.Signature):
    \"\"\"{role_system_message.format(name=name, responsibilities=profile)}\"\"\"

    history: str = dspy.InputField()
    current_order: str = dspy.InputField()
    response: str = dspy.OutputField()
    
{name.lower()} = dspy.Predict({class_name})
    """

    # Return the class definition string
    return class_template.strip()
    
# Generate the class definitions
template = ["import dspy"]
agents_dict_template = "agents = {\n"
for e in profiles:
    name = e['name']
    template.append(build_dspy_signature(name, e['profile']))
# build a dict of name: dspy.Predict
    agents_dict_template += f'    "{name.lower()}": {name.lower()},\n'
agents_dict_template += "}\n"

template.append(agents_dict_template)
    
with open('agent_pool_dspy.py', 'w') as f:
    f.write('\n\n'.join(template))