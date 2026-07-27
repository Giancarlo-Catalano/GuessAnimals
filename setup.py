import json
import os
import sys

import numpy as np

import utils
from sentence_transformers import SentenceTransformer

CACHED_DATA_FOLDER = "prepared_data"
LIST_OF_ANIMALS_FILE = os.path.join(CACHED_DATA_FOLDER, "list_of_animals.json")
ANIMAL_SIMILARITIES_FILE = os.path.join(CACHED_DATA_FOLDER, "animal_similarities.npy")

URL_ANIMALS = r"https://gist.github.com/raineorshine/599777e98e5e968a15c545043973f035/raw"

def download_list_of_animals():
    downloaded_list_of_animals = utils.download_raw_text_from_website(URL_ANIMALS)

    utils.log(downloaded_list_of_animals)
    # it's a single string of capitalised animals
    animals = downloaded_list_of_animals.split("\n")

    animals = [animal.lower().strip() for animal in animals]
    animals.sort()

    # Write the list of animals to JSON


    with open(LIST_OF_ANIMALS_FILE, "w+") as file:
        json.dump(animals, file)
    utils.log(f"Saved list of animals to {LIST_OF_ANIMALS_FILE}")

    return animals

def get_similarity_matrix_between_animals(animals: list[str]):
    # Prepare EmbeddingGemma
    DEVICE = "mps"
    MODEL_ID = "google/embeddinggemma-300M"
    model: SentenceTransformer = SentenceTransformer(MODEL_ID, device=DEVICE)

    # Run inference with queries and documents
    embeddings = model.encode(animals)
    similarities: np.ndarray = model.similarity(embeddings, embeddings).numpy()

    utils.log(f"The similarities is of type {type(similarities)}")
    utils.log(f"The similarities is has shape {similarities.shape}")

    np.save(ANIMAL_SIMILARITIES_FILE, similarities )
    utils.log(f"Saved list of animals to {LIST_OF_ANIMALS_FILE}")




def setup():
    utils.log("Debug setup")

    if False:
        animals = download_list_of_animals()

    # temp code
    animals = json.load(open("prepared_data/list_of_animals.json", "r"))

    # make the similarity matrix
    get_similarity_matrix_between_animals(animals)












