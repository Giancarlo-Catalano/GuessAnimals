# What is this repo?

I love word games, and I sometimes play things like [Hot and Cold](https://hotandcoldgame.com/), where the main idea is that between any pair of words, there is a similarity metric.

I always wondered what is the ideal set of guesses in order to get a quick idea, i.e:

> Given N guesses, which words will guarantee that the secret word is close to one of them? 

I am interested in word embeddings, and I just discovered that I can run an embedder (for sentences) locally on my modest MacBook M1 Air.

Click [here](#results) to jump to the results.

# Methodology
It's hard to define when words are "close enough". 
I don't need "close enough", I just need "as close as I can get". I decided to use this (the last function) as a **fitness metric**:

```python
all_words = {"dog", "cat", "ant", "bee"} # I will use a set of animals

def to_vector(word: str) -> Embedding:
    pass # use EmbeddingGemma, produces a Tensor

def similarity(vec_1: Embedding, vec_2: Embedding) -> float:
    pass # cosine similarity! But it can be changed easily

# this is the important part!
def quality_metric(chosen_words: set[str]) -> float:
    def closest_similarity_for_word(word:str) -> float:
        return max(similarity(to_vector(word), chosen_word)
                   for chosen_word in chosen_words)
    
    not_chosen_words = all_words.difference(chosen_words)
    return min(closest_similarity_for_word(word)
               for word in not_chosen_words)
```
In better terms:
1. For a given word, I see which word [from my set] it is closest to. I record the similarity, and bigger is better (cosine similarity!)
2. In all the words that I didn't pick already, what is the **worst** closest distance?

If you are interested in how this is implemented, you can look at `objective.py`, where I define `make_maxmin_distance_objective`. 
Note that the objective is multiplied by -1 so that it becomes a _minimisation_ task (which is less confusing to me). 

If we have a quality metric, then we can do **optimisation** via a Genetic Algorithm. 


> [!NOTE]
> Interesting detail: just getting the cosine similarity between the animals did not work well, because the words can have more than one meaning.
> For example, I kept seeing that `fowl` is similar to `catfish`, which makes sense but not for animals. 
> Instead, I find the similarity between `The animal fowl` and `The animal catfish`. I'm sure this could be much more sophisticated but I'm happy with the results. 



## Metaheuristic process
> [!NOTE]
> Feel free to skip this section if you don't care about Metaheuristics / Search methods / Genetic Algorithms / Nature Based Methods / Non-Deterministic Solvers / Evolutionary Computing.

We aim to find the set of words which achieves the maximum of the fitness metric defined above. 
Then:
* What is an individual? It will be a set of words, fixed to a certain size. This size is set as a static parameter, which I call N.
  * For speed, I store the _index_ of each word, rather than the word itself.
* What are the operators? I *could* use normal operators for a set of items, but I want to try again with what I implemented in [this paper](https://dl.acm.org/doi/10.1007/978-3-032-11442-6_3). 
  * Mutation: We swap items for similar items. Items are similar based on cosine similarity, and this similarity informs a Markov transition matrix. 
  * Crossover: Offspring are guaranteed the intersection of their parents
  * Selection: Tournament selection

I will use a traditional mu = lambda algorithm, with mechanisms to avoid duplicates. Note that I did not use a library for this, because PyMoo is not the that great, and the other libraries are quite poor IMHO.

## Speed! And setup
Yes, I can run EmbeddingGemma locally, but it's slow.
It's easier to run it once, and obtain all the similarities.
Once I do that, I don't even need to use the embeddings, or cosine similarity!!!


Due to this, there is a setup script (setup.py) which will:
1. download a list of animals from a URL
2. Associate each animal with an index
3. setup EmbeddingGemma, and get similarities between all the animals
4. create a matrix representing the similarity between every animal **index**
5. save that matrix using numpy.

For your convenience, I will save a copy of my fetched data in the repo, so you don't need to run any of this!.

If you want to run the setup, set the variable `RUN_SETUP` to `True` in main, and add a file called secrets.json, with the attribute `GOOGLE_GEMINI_API_KEY` being populated with your API key. 

## Why not other methods?
There are other options we could have considered:
* cluster all the words, using the inverse of the similarity as a distance metric. The centroids are our desired set.
  * --> That could work, but it does not guarantee that the worst distance is low... The clusters could be huge!
* Given the set for N-1, greedily find the new word that improves the objective the most. 
  * --> That might work, but we assume that the problem is easy. I think this GA option allows us to directly state our intention, and the algorithm just searches without any assumptions
  * --> Note how my results change completely for every N!

# Results

| N   | Animals                                                                                         |
|-----|-------------------------------------------------------------------------------------------------|
| 1   | portuguese man o' war                                                                           |
| 2   | sea snail, woodpecker                                                                           |
| 3   | panda, sturgeon, water buffalo,                                                                 |
| 4   | bee, cockroach, giant panda, sugar glider                                                       |
| 5   | beetle, buffalo, gull, panda, quokka                                                            |
| 6   | ant, bug, dormouse, giant panda, guppy, wolverine                                               |
| 7   | bat, giant panda, hyena, mite, spider, swordfish, tasmanian devil                               |
| 8   | basilisk, bobcat, cockroach, gamefowl, jaguar, panda, panthera hybrid, termite                  |
| 9   | carp, fowl, gazelle, ground shark, land snail, panda, siamese fighting fish, tarantula, tarsier |
| 10  | cephalopod, damselfly, hookworm, kangaroo, lynx, panda, partridge, salamander, scallop, skink   |

## Dependencies
You will need the following libraries:
* (to run the setup)
  * sentence_transformers
* To run the metaheuristic
  * numpy

## Known issues
* The list of animals that I use is not great, because it contains many animals that are either identical (e.g. snail vs land snail), or that are categories (canid, felidae), or some animals are obscure
* GA code is not as fast as it could be, on my machine it takes ~ 5 seconds per search.
* I could use other lists of items (i.e. Jobs), but I care more about the procedure than the results...
* The results are definitely skewed towards mammals. I suspect this is because most of my dataset is mammals...