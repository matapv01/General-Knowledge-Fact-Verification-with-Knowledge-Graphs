examples = [
    {
        "query": "Israeli Australian had a religion.",
        "divided": [
            "Israeli Australian had a religion"
        ],
        "graph": [
            ('Israeli Australian', 'had a religion', 'UNKNOWN religion 1')
        ]
    },
    {
        "query": "Agra Airport is located in Uttar Pradesh part of Lào Cai Province and Bundelkhand.",
        "divided": [
            "Agra Airport is located in Uttar Pradesh",
            "Uttar Pradesh is part of Lào Cai Province",
            "Uttar Pradesh is part of Bundelkhand"
        ],
        "graph": [
            ('Agra Airport', 'located in', 'Uttar Pradesh'),
            ('Uttar Pradesh', 'part of', 'Lào Cai Province'),
            ('Uttar Pradesh', 'part of', 'Bundelkhand')
        ]
    },
    {
        "query": "The celestial body known as 1101 Clematis has an escape velocity of 0.02 k.p.s., a temperature of 155 kelvins and an apoapsis of 520906000.0 kilometres.",
        "divided": [
            "The celestial body known as 1101 Clematis has an escape velocity of 0.02 k.p.s.",
            "The celestial body known as 1101 Clematis has a temperature of 155 kelvins",
            "The celestial body known as 1101 Clematis has an apoapsis of 520906000.0 kilometres"
        ],
        "graph": [
            ('1101 Clematis', 'escape velocity', '"0.02"'), 
            ('1101 Clematis', 'temperature', '"155"'), 
            ('1101 Clematis', 'apoapsis', '"5.20906E8"')
        ]
    },
    {
        "query": "103 Colmore Row was not designed by the architect William T. Leighton, his hometown wasn't Birmingham.",
        "divided": [
            "103 Colmore Row was not designed by the architect William T. Leighton",
            "William T. Leighton's hometown wasn't Birmingham"
        ],
        "graph": [
            ('103 Colmore Row', 'designed by', 'UNKNOWN architect 1'),
            ('UNKOWN architect 1', 'hometown', 'Birmingham')
        ]
    },
    {
        "query": "His name is Robert M. Gray and he won an award for it.",
        "divided": [
            "Robert M. Gray won an award"
        ],
        "graph": [
            ('Robert M. Gray', 'won the award', 'UNKNOWN award 1')
        ]
    },
    {
        "query": "Are you familiar with Brandon Penn? He attended college!",
        "divided": [
            "Brandon Penn attended college"
        ],
        "graph": [
            ('Brandon Penn', 'attended the college', 'UNKNOWN college 1')
        ]
    },
    {
        "query": "There was a garrison at Allenby Formation.",
        "divided": [
            "There was a garrison at Allenby Formation"
        ],
        "graph": [
            ('UNKNOWN garrison', 'the garrison at', 'Allenby Formation')
        ]
    },
    {
        "query": "Paulo Rosales has had a youth club.",
        "divided": [
            "Paulo Rosales has had a youth club"
        ],
        "graph": [
            ('Paulo Rosales', 'has had a youth club', 'UNKNOWN youth club 1')
        ]
    },
    {
        "query": "William Anders nationality is the United States, but he was born in Schaumburg, Illinois, US.",
        "divided": [
            "William Anders nationality is the United States",
            "William Anders was born in Schaumburg, Illinois, US"
        ],
        "graph": [
            ('William Anders', 'nationality', 'United States'),
            ('William Anders', 'born in', 'Schaumburg, Illinois, US')
        ]
    },
    {
        "query": "I have, Colleen Howe had a child.",
        "divided": [
            "Colleen Howe had a child"
        ],
        "graph": [
            ('Colleen Howe', 'had a child', 'UNKNOWN child 1')
        ]
    },
    {
        "query": "Alfons Gorbach's birthplace is Weleetka, Oklahoma, in Austria-Hungary.",
        "divided": [
            "Alfons Gorbach's birthplace is Weleetka, Oklahoma, Austria-Hungary"
        ],
        "graph": [
            ('Alfons Gorbach', 'birthplace', 'Weleetka, Oklahoma, Austria-Hungary')
        ]
    },
    {
        "query": "Bob MacMillan, member of the Parti Pesaka Bumiputera Bersatu party, resides in Demak Jaya, Jalan Bako, Kuching, Sarawak.",
        "divided": [
            "Bob MacMillan is the member of the Parti Pesaka Bumiputera Bersatu party",
            "Bob MacMillan resides in Demak Jaya, Jalan Bako, Kuching, Sarawak"
        ],
        "graph": [
            ('Bob MacMillan', 'member of', 'Parti Pesaka Bumiputera Bersatu party'),
            ('Bob MacMillan', 'resides in', 'Demak Jaya, Jalan Bako, Kuching, Sarawak')
        ]
    }
]


def get(query, shot=12):
    prompt = """You need to segment the given query then extract the potential knowledge graph structures.

Notes)
1). Use the original description in the query with enough context, NEVER use unspecific words like 'in', 'is', 'for', 'of', 'have', 'go to', etc.
2). For nodes or relations that are unknown, you can use the keyword 'UNKNOWN' with a unique ID, e.g., 'UNKNOWN artist 1', 'UNKNOWN relation 1'.
3). For statements with negations, such as 'not', 'wasn't', 'didn't', you should use the keyword 'UNKNOWN' for the negated nodes, e.g., 'A does not live in B' results in the triple ('A', 'live in', 'UNKNOWN location 1').
4). For values without textual semantic meanings, such as numbers, heights and speeds, you should only preserve the value itself with double quotes without any units, e.g., '"156"'. For large numbers with lots of 0000, use the scientific notation, e.g., '"5.2E06"'.
5). Return the segmented query and extracted graph structures strictly following the format:
    {
        "divided": [
            "segment 1",
            ...
        ],
        "graph": [
            ('head', 'relation', 'tail'),
            ...
        ]
    }
6). NEVER provide extra descriptions or explanations, such as something like 'Here is the extracted knowledge graph structure'.

Examples)
"""

    for i in range(min(shot, len(examples))):
        prompt += f"""
{i+1}. query: '{examples[i]['query']}'
{{
    "divided": {examples[i]['divided']},
    "graph": {examples[i]['graph']}
}}
"""

    return prompt + f"""
Your task)
**Read and follow the instructions and examples step by step**
query: '{query}'
"""
        
