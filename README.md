# What is this repo?

I love word games, and I sometimes play things like [Hot and Cold](https://hotandcoldgame.com/), where the main idea is that between any pair of words, there is a similarity metric.

I always wondered what is the ideal set of guesses in order to get a quick idea, i.e:

> Given N guesses, which words will guarantee that the secret word is close to one of them? What set of words is a "uniformly distributed" within the space of words?

We have word embeddings now, and I just discovered that I can run an embedder (for sentences) locally on my modest MacBook M1 Air. 

# Methodology
It's hard to define when words are "close enough", especially using embedded vectors. 
I don't need "close enough", I just need a quality metric. I decided to use this:

```python
all_words = {"dog", "cat", "ant", "bee"} # I will use a set of animals

def to_vector(word: str) -> Embedding:
    pass # use EmbeddingGemma, produces a Tensor

def distance(vec_1: Embedding, vec_2: Embedding) -> float:
    pass # cosine similarity! But it can be changed easily

# this is the important part!
def quality_metric(chosen_words: set[str]) -> float:
    def closest_distance_for_word(word:str) -> float:
        return min(distance(to_vector(word), chosen_word)
                   for chosen_word in chosen_words)
    
    not_chosen_words = all_words.difference(chosen_words)
    return max(closest_distance_for_word(word)
               for word in not_chosen_words)
```
In better terms:
1. For a given word, I see which word [from my set] it is closest to. I record the distance, and smaller is better.
2. In all the words that I didn't pick already, which word has the **worst** closest distance?


If we have a quality metric, then we can do **optimisation**. 

## Metaheuristic process
> [!NOTE]
> Feel free to skip this section if you don't care about Metaheuristics / Search methods / Genetic Algorithms / Nature Based Methods / Non-Deterministic Solvers / Evolutionary Computing.

We aim to find the set of words which achieves the maximum of the fitness metric defined above. 
Then:
* What is an individual? It will be a set of words, fixed to a certain size. This size is set as a static parameter.
* What are the operators? I *could* use normal operators for a set of items, but I want to try again with what I implemented in [this paper](https://dl.acm.org/doi/10.1007/978-3-032-11442-6_3). 
  * Mutation: We swap items for similar items. Items are similar based on cosine similarity
  * Crossover: Offspring are guaranteed the intersection of their parents
  * Selection: Tournament selection, size = 3 (as is standard...)

I will use a traditional mu = lambda algorithm, nothing special there.

## Optimisation
Yes, I can run EmbeddingGemma locally, but it's slow.
It's easier to run it once, and obtain all the similarities.
Once I do that, I don't even need to use the embeddings, or cosine similarity!!!

Due to this, there is a setup script (setup.py) which will:
* download a list of animals from a URL
* Associate each animal with an index
* setup EmbeddingGemma, and get similarities between all the animals
* create a matrix representing the similarity between every animal **index**
* save that matrix using numpy.

For your convenience, I will save a copy here. 

# RESULTS
> [!IMPORTANT]
> I have not gotten here yet! I'm writing the readme first, on 27/07/2026

