import json
import os
import sys
import utils
URL_ANIMALS = r"https://gist.github.com/raineorshine/599777e98e5e968a15c545043973f035/raw"


def get_list_of_animals():
    downloaded_list_of_animals = utils.download_raw_text_from_website(URL_ANIMALS)

    utils.log(downloaded_list_of_animals)
    # it's a single string of capitalised animals
    animals = downloaded_list_of_animals.split("\n")

    animals = [animal.lower().strip() for animal in animals]
    animals.sort()

    # Write the list of animals to JSON
    CACHED_DATA_FOLDER = "prepared_data"
    LIST_OF_ANIMALS_FILE = os.path.join(CACHED_DATA_FOLDER, "list_of_animals.json")

    with open(LIST_OF_ANIMALS_FILE, "w+") as file:
        json.dump(animals, file)
    utils.log(f"Saved list of animals to {LIST_OF_ANIMALS_FILE}")

    return animals

def get_similarity_matrix_between_animals():
    pass

def setup():
    utils.log("Debug setup")

    if False:
        animals = get_list_of_animals()
        print(animals)










