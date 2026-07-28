import json
import os
import sys

import numpy as np

import utils


CACHED_DATA_FOLDER = "prepared_data"
LIST_OF_ANIMALS_FILE = os.path.join(CACHED_DATA_FOLDER, "list_of_animals.json")
ANIMAL_SIMILARITIES_FILE = os.path.join(CACHED_DATA_FOLDER, "animal_similarities.npy")

URL_ANIMALS = r"https://gist.github.com/raineorshine/599777e98e5e968a15c545043973f035/raw"
# the URL below has more animals
#URL_ANIMALS = r"https://raw.githubusercontent.com/skjorrface/animals.txt/refs/heads/master/animals.txt"

def download_list_of_animals():
    downloaded_list_of_animals = utils.download_raw_text_from_website(URL_ANIMALS)

    utils.log(downloaded_list_of_animals)
    # it's a single string of capitalised animals
    animals = downloaded_list_of_animals.split("\n")

    animals = [animal.lower().strip() for animal in animals]
    animals = [animal for animal in animals if len(animal)>2]  # remove empty animals
    animals.sort()

    # Write the list of animals to JSON


    with open(LIST_OF_ANIMALS_FILE, "w+") as file:
        json.dump(animals, file)
    utils.log(f"Saved list of animals to {LIST_OF_ANIMALS_FILE}")

    return animals

def save_similarity_matrix(animals: list[str]):
    from sentence_transformers import SentenceTransformer
    # I import here because this is a slow import, I don't want to always do it!

    # Prepare EmbeddingGemma
    DEVICE = "mps"
    MODEL_ID = "google/embeddinggemma-300M"
    model: SentenceTransformer = SentenceTransformer(MODEL_ID, device=DEVICE)

    # Run inference with queries and documents
    better_query_for_animals = [f"The animal {animal}" for animal in animals]
    embeddings = model.encode(better_query_for_animals)
    similarities: np.ndarray = model.similarity(embeddings, embeddings).numpy()

    utils.log(f"The similarities is of type {type(similarities)}")
    utils.log(f"The similarities is has shape {similarities.shape}")

    np.save(ANIMAL_SIMILARITIES_FILE, similarities )
    utils.log(f"Saved list of animals to {LIST_OF_ANIMALS_FILE}")




def setup():
    animals = download_list_of_animals()

    # make the similarity matrix
    save_similarity_matrix(animals)












