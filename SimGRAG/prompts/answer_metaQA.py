examples = [
    {
        "query": "the films that share actors with the film Dil Chahta Hai were released in which years?",
        "evidences": [
            [
                ("Dil Chahta Hai", "starred_actors", "Aamir Khan"),
                ("Ghajini", "release_year", "2008"),
                ("Ghajini", "starred_actors", "Aamir Khan"),
            ],
            [
                ("Dil Chahta Hai", "written_by", "Farhan Akhtar"),
                ("Luck by Chance", "starred_actors", "Farhan Akhtar"),
                ("Luck by Chance", "release_year", "2009"),
            ]
        ],
        "answer": "According to graph [1], the film was released in 2008. Graph [2] is not useful."
    },
    {
        "query": "who are the directors of the movies written by the writer of The Green Mile?",
        "evidences": [
            [
                ("The Green Mile", "written_by", "Stephen King"),
                ("The Shawshank Redemption", "directed_by", "Frank Darabont"),
                ("The Shawshank Redemption", "written_by", "Stephen King"),
            ],
            [
                ("The Shawshank Redemption", "directed_by", "Frank Darabont"),
                ("The Mangler", "written_by", "Stephen King"),
                ("The Mangler", "written_by", "Tobe Hooper"),
            ]
        ],
        "answer": "According to graph [1], the director is Frank Darabont. Graph [2] is not useful."
    },
    {
        "query": "which person directed the films acted by the actors in Jawbreaker?",
        "evidences": [
            [
                ("Visioneers", "starred_actors", "Judy Greer"),
                ("Jawbreaker", "starred_actors", "Judy Greer"),
                ("Visioneers", "directed_by", "Jared Drake"),
            ],
            [
                ("Jawbreaker", "starred_actors", "Judy Greer"),
                ("Gate of Hell", "starred_actors", "Kazuo Hasegawa"),
                ("Gate of Hell", "directed_by", "Teinosuke Kinugasa"),
            ]
        ],
        "answer": "According to graphs [1][2], the person can be Jared Drake or Teinosuke Kinugasa."
    },
    {
        "query": "who is listed as director of the movies starred by December Boys actors?",
        "evidences": [
            [
                ("December Boys", "starred_actors", "Daniel Radcliffe"),
                ("Harry Potter and the Chamber of Secrets", "starred_actors", "Daniel Radcliffe"),
                ("Harry Potter and the Chamber of Secrets", "directed_by", "Chris Columbus"),
            ],
            [
                ("December Boys", "written_by", "Marc Rosenberg"),
                ("Dingo", "written_by", "Marc Rosenberg"),
                ("Dingo", "directed_by", "Rolf de Heer"),
            ]
        ],
        "answer": "According to graph [1], the director is Chris Columbus. Graph [2] is not useful."
    },
    {
        "query": "what types are the films directed by the director of For Love or Money?",
        "evidences": [
            [
                ("For Love or Money", "directed_by", "Barry Sonnenfeld"),
                ("Addams Family Values", "directed_by", "Barry Sonnenfeld"),
                ("Addams Family Values", "has_genre", "Comedy"),
            ],
            [
                ("Big Trouble", "directed_by", "Barry Sonnenfeld"),
                ("Big Trouble", "directed_by", "John Cassavetes"),
                ("Great Directors", "directed_by", "Angela Ismailos")
            ]
        ],
        "answer": "According to graph [1], the type is Comedy. Graph [2] is not useful."
    },
    {
        "query": "who wrote films that share actors with the film Anastasia?",
        "evidences": [
            [
                ("Anastasia", "starred_actors", "Ingrid Bergman"),
                ("Spellbound", "starred_actors", "Ingrid Bergman"),
                ("Spellbound", "written_by", "Ben Hecht"),
            ],
            [
                ("Anastasia", "starred_actors", "John Cusack"),
                ("Floundering", "starred_actors", "John Cusack"),
                ("Floundering", "written_by", "Peter McCarthy"),
            ]
        ],
        "answer": "According to graphs [1][2], the writter is Ben Hecht or Peter McCarthy."
    },
    {
        "query": "the movies that share actors with the movie The Constant Nymph were released in which years?",
        "evidences": [
            [
                ("The Constant Nymph", "starred_actors", "Joan Fontaine"),
                ("This Above All", "starred_actors", "Joan Fontaine"),
                ("This Above All", "release_year", "1942"),
            ],
            [
                ("The Constant Gardener", "starred_actors", "Ralph Fiennes"),
                ("Land of the Blind", "starred_actors", "Ralph Fiennes"),
                ("Land of the Blind", "release_year", "2006")
            ]
        ],
        "answer": "According to graph [1], the movie was released in 1942. Graph [2] is not useful."
    },
    {
        "query": "when did the movies release whose actors also appear in the movie Cast a Deadly Spell?",
        "evidences": [
            [
                ("Cast a Deadly Spell", "starred_actors", "Julianne Moore"),
                ("The Shipping News", "release_year", "2001"),
                ("The Shipping News", "starred_actors", "Julianne Moore"),
            ],
        ],
        "answer": "According to graph [1], the movie was released in 2001."
    },
    {
        "query": "what languages are the films that share directors with The Age of Innocence in?",
        "evidences": [
            [
                ("The Age of Innocence", "directed_by", "Martin Scorsese"),
                ("Kundun", "directed_by", "Martin Scorsese"),
                ("Kundun", "in_language", "Tibetan"),
            ],
            [
                ("The Age of Innocence", "directed_by", "Martin Scorsese"),
                ("My Voyage to Italy", "starred_actors", "Martin Scorsese"),
                ("My Voyage to Italy", "in_language", "Italian")
            ]
        ],
        "answer": "According to graph [1], the language is Tibetan. Graph [2] is not useful."
    },
    {
        "query": "who are the actors in the movies written by the writer of Confessions of a Teenage Drama Queen?",
        "evidences": [
            [
                ("Confessions of a Teenage Drama Queen", "written_by", "Gail Parent"),
                ("The Main Event", "written_by", "Gail Parent"),
                ("The Main Event", "starred_actors", "Barbra Streisand"),
            ],
            [
                ("Confessions of a Teenage Drama Queen", "directed_by", "Sara Sugarman"),
                ("Very Annie Mary", "written_by", "Sara Sugarman"),
                ("Very Annie Mary", "starred_actors", "Jonathan Pryce"),
            ]
        ],
        "answer": "According to graph [1], the actor is Barbra Streisand. Graph [2] is not useful."
    },
    {
        "query": "who starred in the films whose directors also directed The Decline of the American Empire?",
        "evidences": [
            [
                ("The Decline of the American Empire", "directed_by", "Denys Arcand"),
                ("Stardom", "directed_by", "Denys Arcand"),
                ("Stardom", "starred_actors", "Jessica Paré"),
            ],
        ],
        "answer": "According to graph [1], Jessica Paré starred in the films."
    },
    {
        "query": "the films that share directors with the film Déjà Vu were released in which years",
        "evidences": [
            [
                ("Déjà Vu", "directed_by", "Anthony B. Richmond"),
                ("Anthony B. Richmond", "directed_by", "Déjà Vu"),
                ("Déjà Vu", "release_year", "1985")
            ],
            [
                ("Déjà Vu", "directed_by", "Henry Jaglom"),
                ("Henry Jaglom", "directed_by", "A Safe Place"),
                ("A Safe Place", "release_year", "1971")
            ],
            [
                ("Déjà Vu", "directed_by", "Anthony B. Richmond"),
                ("Anthony B. Richmond", "directed_by", "Déjà Vu"),
                ("Déjà Vu", "release_year", "1997")
            ]
        ],
        "answer": "According to graphs [1][2][3], the films were released in 1985, 1971 or 1997."
    },
]


def get(query, evidences, shot=12):
    prompt = """Please answer the question based on the given evidences from a knowledge graph. 

Notes)

1). Use the original text in the valid evidences as answer output, NEVER rephrase or reformat them.
2). There may be different answers for different evidences. Return all possible answer for every evidence graph, except for those that are obviously not aligned with the query.
3). You should provide a brief reason with several words, then tell all the answers you found.

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
