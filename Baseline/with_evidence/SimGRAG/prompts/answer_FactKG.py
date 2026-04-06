examples = [
    {
        "query": "Mick Walker (footballer, born 1940) is the leader of 1993–94 Notts County F.C. season.",
        "evidences": [
            [
                ('Mick Walker (footballer, born 1940)', 'manager', '1993–94 Notts County F.C. season'),
                ('Mick Walker (footballer, born 1940)', 'birthDate', '"1940-11-27"')
            ],
            [
                ('Mick Walker (footballer, born 1940)', 'manager', '1994–95 Notts County F.C. season'),
                ('Mick Walker (footballer, born 1940)', 'birthDate', '"1940-11-27"')
            ],
            [
                ('Mick Walker (footballer, born 1940)', 'manager', '1992–93 Notts County F.C. season'),
                ('Mick Walker (footballer, born 1940)', 'birthDate', '"1940-11-27"')
            ]
        ],
        "answer": "As graphs [1][2][3] say that Mick Walker is the manager but not the leader, the answer is False"
    },
    {
        "query": "It was a parent company of Memorial Press Group.",
        "evidences": [
            [
                ('Memorial Press Group', 'parent', 'Enterprise NewsMedia, 1998-2006')
            ],
            [
                ('Memorial Press Group', 'parent', 'GateHouse Media, 2006')
            ],
            [
                ('Memorial Press Group', 'parent', 'Prescott Publishing, 1979-1998')
            ]
        ],
        "answer": "As graph [1] says that Enterprise NewsMedia, 1998-2006 is the parent company of Memorial Press Group, the answer is True"
    },
    {
        "query": "The ship, completed April 6, 2005, weighs 1850 tons and is 125800.0 mm in length.",
        "evidences": [
            [
                ('"2005-04-06"', 'completionDate', 'A-Rosa_Luna'),
                ('A-Rosa_Luna', 'shipDisplacement', '"1850.0"'),
                ('A-Rosa_Luna', 'length', '"125.8"')
            ]
        ],
        "answer": "As graph [1] says that A-Rosa_Luna was completed on 2005-04-06, weighs 1850 tons and is 125.8 meters in length, the answer is True"
    },
    {
        "query": "A Night at Boomers, Vol. 2 is signed to RCA Records whose distributor is Sony Music Entertainment.",
        "evidences": [
            [
                ('Greatest_Hits,_Vol._2_(Ronnie_Milsap_album)', 'label', 'RCA_Records'),
                ('RCA_Records', 'distributor', 'Sony Music Entertainment')
            ]
        ],
        "answer": "As graph [1] is about Greatest_Hits,_Vol._2_(Ronnie_Milsap_album) but not A Night at Boomers, Vol. 2, the answer is False"
    },
    {
        "query": "Honda makes the Honda CB125 which has the Honda J engine type.",
        "evidences": [
            [
                ('Honda', 'manufacturer', 'Honda_CB125'),
                ('Honda_CB125', 'manufacturer', 'Honda'),
            ]
        ],
        "answer": "As graph [1] does not mention the engine type, the answer is False"
    },
    {
        "query": "Let Go (Susie Luchsinger album) is signed to a record label wherein Universal Music group is the parent company.",
        "evidences": [
            [
                ('Let_Loose_(album)', 'label', 'Mercury_Records'),
                ('Mercury_Records', 'parent', 'Universal_Music_Group')
            ]
        ],
        "answer": "As graph [1] is about Let_Loose_(album) but not Let Go (Susie Luchsinger album), the answer is False"
    },
    {
        "query": "A city lies 214 metres above sea level and has a population density of 47.14307355803265 people for each of its 140.8 square kilometres.",
        "evidences": [
            [
                ('"214"', 'elevationM', 'Auburn,_Alabama'),
                ('Auburn,_Alabama', 'populationDensity', '"1.530682973208576E8"'),
                ('Auburn,_Alabama', 'areaTotal', '"140.8"')
            ]
        ],
        "answer": "As graph [1] says that Auburn, Alabama has a population density of 1.530682973208576E8 but not 47.14307355803265, the answer is False"
    },
    {
        "query": "Agreed, Adlai Stevenson I was the Vice President.",
        "evidences": [
            [
                ('Adlai_Stevenson_I', 'vicePresident', 'Grover_Cleveland')
            ],
            [
                ('Adlai_Stevenson_I', 'vicepresident', 'Grover_Cleveland')
            ],
            [
                ('Adlai_Stevenson_I', 'president', 'Grover_Cleveland')
            ]
        ],
        "answer": "As graph [1] says that Adlai Stevenson I was the Vice President, the answer is True"
    },
    {
        "query": "Well I know that Sam's Choice has a parent company.",
        "evidences": [
            [
                ('Sam\'s_Choice', 'related', 'Bubba_Cola')
            ],
            [
                ('Sam\'s_Choice', 'origin', 'United_States')
            ],
            [
                ('Sam\'s_Choice', 'type', 'Private_label')
            ]
        ],
        "answer": "As all graphs are not about its parent company, the answer is False"
    },
    {
        "query": "Have you ever heard of Hortensio Quijano? He was Vice President.",
        "evidences": [
            [
                ('Hortensio_Quijano', 'keyPerson', 'List_of_Vice_Presidents_of_Argentina')
            ],
            [
                ('Hortensio_Quijano', 'title', '"Peronist Party nominee for Vice President of Argentina and President of the Argentine Senate"')
            ],
            [
                ('Academia_Mexicana_de_Derechos_Humanos__1', 'title', '"Vice-President"')
            ]
        ],
        "answer": "As graphs [1][2] says that Hortensio Quijano was Vice President, the answer is True"
    },
    {
        "query": "The leader of Rushin' Ballet is Gordon Douglas the director!",
        "evidences": [
            [
                ('Rushin\' Ballet', 'director', 'Gordon_Douglas_(director)'),
                ('Gordon_Douglas_(director)', 'occupation', 'Film_director')
            ],
            [
                ('Rushin\'Ballet', 'director', 'Gordon_Douglas_(director)'),
                ('Gordon_Douglas_(director)', 'occupation', '"Film director"')
            ],
            [
                ('Rushin\'Ballet', 'director', 'Gordon_Douglas_(director)'),
                ('Gordon_Douglas_(director)', 'occupation', 'Gordon_Douglas_(director)__1')
            ]
        ],
        "answer": "As graphs [1][2][3] do not mention that Gordon Douglas is the leader, the answer is False"
    },
    {
        "query": "His name is Juan March Ordinas and he has an award.",
        "evidences": [
            [
                ('Juan_March_Ordinas', 'alias', '"Joan March i Ordinas"'),
            ],
            [
                ('Juan_March_Ordinas', 'birthName', '"Juan March Ordinas"'),
            ],
        ],
        "answer": "As none of the graphs mention an award, the answer is False"
    },
]


def get(query, evidences, shot=12):
    prompt = """Please verify the statement based on the given evidences from a knowledge graph. 

Notes)

1). If there is any evidence that completely supports the statement, the answer is 'True', otherwise is 'False'.
2). Be careful for the relations in evidences. For example, 'builder' does not equal to 'owner', 'manager' does not equal to 'leader'.
3). For questions like 'A has a wife', if there is any evidence that A has a spouse with any name, the answer is 'True'.
4). When no evidence is provided, you MUST verify the statement and return 'True' or 'False' based on your common sense. NEVER say ANYTHING like 'Please provide more evidences'.
5). You should provide a brief reason with several words, then tell that the answer is 'True' or 'False'. NEVER say ANYTHING without direct reason like 'Based on the provided evidences, ...', 'Here is my reasoning: ...', etc.

Examples)
"""

    for i in range(min(shot, len(examples))):
        prompt += f"""
{i+1}. query: '{examples[i]['query']}'
    evidences: {", ".join([f"graph [{j+1}]: {evidence}" for j, evidence in enumerate(examples[i]['evidences'])])}
    answer: '{examples[i]['answer']}'
"""

    return prompt + f"""
Your task)
**Read and follow the instructions and examples step by step**
query: '{query}'
evidences: {", ".join([f"graph [{j+1}]: {evidence}" for j, evidence in enumerate(evidences)])}
"""
