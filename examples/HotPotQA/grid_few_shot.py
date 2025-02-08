import cognify
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.llm import Demonstration


generate_query_0_few_shots = [
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("question"), value= "Juan Pablo Di Pace had a role on the sequel series to what Jeff Franklin-created American sitcom?"),
        ],
        output="Juan Pablo Di Pace role sequel series Jeff Franklin American sitcom",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("question"), value= " The Badge is a 2002 mystery-thriller film starring an American actress whose notible films include what 1999 Martin Scorsese film?"),
        ],
        output="The Badge 2002 film American actress notable films 1999 Martin Scorsese",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("question"), value= "When was the Houston Rockets point guard recruited by Frank Sullivan born?"),
        ],
        output="To find the birth date of the Houston Rockets point guard recruited by Frank Sullivan, you’d typically need reliable context or references.",
    ),
]

generate_query_1_few_shots = [
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="['Sharon Watts | Sharon Anne Mitchell (also Watts and Rickman) is a fictional character from the BBC One soap opera \"EastEnders\", played by Letitia Dean. Sharon is one of \"EastEnders\"\\' original chara..."),
            cognify.FilledInput(cognify.Input("question"), value="Who created the TV series which had the storyline Sharongate?")
        ],
        output="creators of EastEnders TV series",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="['Formula D (board game) | Formula D (originally published and still also known as Formula D\u00e9) is a board game that recreates formula racing (F1, CART, IRL). It was designed by Eric Randall and Lauren..."),
            cognify.FilledInput(cognify.Input("question"), value="When was the company founded that now publishes Formula D?")
        ],
        output="Asmodée Éditions founding date",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="['Imiliya | Imiliya \u0907\u092e\u093f\u0932\u093f\u092f\u093e is a small town located in the Kapilvastu District, Lumbini, Nepal.', 'Imliya | Imliya is a village in the Bhopal district of Madhya Pradesh, India. It is located in the Hu..."),
            cognify.FilledInput(cognify.Input("question"), value="Imiliya is a town located in what pilgrimage site?")
        ],
        output="Imiliya is a town located in the Kapilvastu District, Lumbini, Nepal, which is an important pilgrimage site in Buddhism.",
    ),
]

generate_answer_few_shots = [
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="['Annette Snell | Annette Snell (March 22, 1945 \u2013 April 4, 1977) was an American rhythm and blues singer who recorded in the 1960s and 1970s. She died in the Southern Airways Flight 242 crash.', 'Sout..."),
            cognify.FilledInput(cognify.Input("question"), value="Singer Annette Snell died in a airplane crash during a forced landing on a highway in what state?")
        ],
        output="Georgia",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="['Juan Pablo Di Pace | Juan Pablo Di Pace (born July 25, 1979) is an Argentine actor, singer and director. Di Pace began his career in United Kingdom, performing in a number of musicals and appearing ..."),
            cognify.FilledInput(cognify.Input("question"), value="Juan Pablo Di Pace had a role on the sequel series to what Jeff Franklin-created American sitcom?")
        ],
        output="Full House",
    ),
    Demonstration(
        filled_input_variables=[
            cognify.FilledInput(cognify.Input("context"), value="[\"Ontario Raiders | The Ontario Raiders were a member of the National Lacrosse League during the 1998 NLL season. The franchise was founded as an expansion team in Hamilton, Ontario, and played their ..."),
            cognify.FilledInput(cognify.Input("question"), value="How much did retired Canadian professional ice hockey player of Albanian origin spend with his cohorts to purchase the Ontario Raiders?")
        ],
        output="I don't have enough information to answer this question.",
    ),
]

agent_few_shots = {
    "generate_query_0": LMFewShot(max_num=2, user_demos=generate_query_0_few_shots, module_name="generate_query_0"),
    "generate_query_1": LMFewShot(max_num=2, user_demos=generate_query_1_few_shots, module_name="generate_query_1"),
    "generate_answer": LMFewShot(max_num=2, user_demos=generate_answer_few_shots, module_name="generate_answer"),
}