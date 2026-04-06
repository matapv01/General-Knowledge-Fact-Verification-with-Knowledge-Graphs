examples = [
    {
        "query": "the films that share directors with the film Catch Me If You Can were in which languages",
        "divided": [
            "the films were in which languages",
            "the films that directed by the directors",
            "the director of the film Catch Me If You Can"
        ],
        "graph": [
            ("UNKNOWN film 1", "in language", "UNKNOWN language 1"),
            ("UNKNOWN director 1", "direct", "UNKNOWN film 1"),
            ("UNKNOWN director 1", "direct", "Catch Me If You Can")
        ]
    },
    {
        "query": "who are film co-writers of Tim Burns",
        "divided": [
            "the film written by Tim Burns",
            "the film also written by any other writer"
        ],
        "graph": [
            ("Tim Burns", "write", "UNKNOWN film 1"),
            ("UNKNOWN writer 1", "write", "UNKNOWN film 1")
        ]
    },
    {
        "query": "what languages are the films that share actors with The Vow in",
        "divided": [
            "what languages are the films in",
            "the films have actors",
            "the actor of the film The Vow"
        ],
        "graph": [
            ("UNKNOWN film 1", "in language", "UNKNOWN language 1"),
            ("UNKNOWN film 1", "have actors", "UNKNOWN actor 1"),
            ("UNKNOWN actor 1", "actor of", "The Vow")
        ]
    },
    {
        "query": "who acted together with Breckin Meyer",
        "divided": [
            "the actors acted in a film",
            "Breckin Meyer also acts in the film",
        ],
        "graph": [
            ("UNKNOWN actor 1", "actor of", "UNKNOWN film 1"),
            ("UNKNOWN film 1", "have actor", "Breckin Meyer")
        ]
    },
    {
        "query": "what are the languages spoken in the movies written by The Beekeeper writers?",
        "divided": [
            "what are the languages spoken in the movies",
            "the movies written by writers",
            "The writers of the movie The Beekeeper"
        ],
        "graph": [
            ("UNKNOWN movie 1", "the language spoken", "UNKNOWN language 1"),
            ("UNKNOWN writer 1", "the write of", "UNKNOWN movie 1"),
            ("UNKNOWN writer 1", "the writer of", "The Beekeeper")
        ]
    },
    {
        "query": "what genres do the films that share directors with Scarlet Street fall under",
        "divided": [
            "what genres do the films fall under",
            "the films that have directors",
            "the directors of the film Scarlet Street"
        ],
        "graph": [
            ("UNKNOWN film 1", "have genre", "UNKNOWN genre 1"),
            ("UNKNOWN film 1", "have directors", "Scarlet Street"),
            ("UNKNOWN director 1", "directe", "Scarlet Street")
        ]
    },
    {
        "query": "who is listed as screenwriter of the films starred by The Business actors",
        "divided": [
            "who is listed as screenwriter of the films",
            "the films starred by an actor",
            "the actor starred in The Business"
        ],
        "graph": [
            ("UNKNOWN film 1", "have screenwriter", "UNKNOWN person"),
            ("UNKNOWN film 1", "has actor", "UNKNOWN actor 1"),
            ("UNKNOWN actor 1", "actor of", "The Business")
        ]
    },
    {
        "query": "the actor in Flashpoint also appears in which films",
        "divided": [
            "the actor in Flashpoint",
            "this actor also appears in another films",
        ],
        "graph": [
            ("UNKNOWN actor 1", "actor of", "Flashpoint"),
            ("UNKNOWN actor 1", "actor of", "UNKNOWN film 1"),
        ]
    },
    {
        "query": "who are co-stars of Gabriel Tigerman",
        "divided": [
            "Gabriel Tigerman is the star actor in a film",
            "who is also the actor of the film"
        ],
        "graph": [
            ("Gabriel Tigerman", "actor of", "UNKNOWN film 1"),
            ("UNKNOWN actor 1", "actor of", "UNKNOWN film 1"),
        ]
    },
    {
        "query": "who is listed as director of Richard Wattis starred movies",
        "divided": [
            "Richard Wattis starred in a film",
            "who is listed as director of this film"
        ],
        "graph": [
            ("Richard Wattis", "actor of", "UNKNOWN movie 1"),
            ("UNKNOWN person 1", "director of", "UNKNOWN movie 1"),
        ]
    }
]


def get(query, shot=12):
    prompt = """You need to segment the given query then extract the potential knowledge graph structures.

Notes)
1). Use the original description in the query with enough context, NEVER use unspecific words like 'in', 'appear in', 'for', 'of' etc. ALWAYS use the active voice instead of the passive voice, e.g., 'A is the actor of B' instead of 'B is acted by A'.
2). For nodes or relations that are unknown, you can use the keyword 'UNKNOWN' with a unique ID, e.g., 'UNKNOWN artist 1', 'UNKNOWN relation 1'.
3). NEVER use relations like 'A act together with B', 'A is the co-writer of B', or 'A share director of B' in the extracted graph. You MUST break them into two segments, such as 'A is the actor/writer/director of UNKNOWN movie' and 'B is the actor/writer/director of UNKNOWN movie', which delivers two triples in the graph.
4). Return the segmented query and extracted graph structures strictly following the format:
    {
        "divided": [
            "segment 1",
            ...
        ],
        "graph": [
            ("head", "relation", "tail"),
            ...
        ]
    }
5). NEVER provide extra descriptions or explanations, such as something like 'Here is the extracted knowledge graph structure'.

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
        
