import random
import json
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel

from create_crop_special_article import CropSpecialArticle, create_human_messages


class Item():
    def __init__(self, kor, eng):
        self.kor = kor
        self.eng = eng

class Article(BaseModel):
    crop: str
    title: str
    body: str
    author: str
    change_rate: float


def set_env():
    global model_name
    global crop_special_prompt_file_name, crops_file_name, authors_file_name, related_elements_file_name, crop_special_results_directory_name

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    model_name = os.getenv("MODEL_NAME")
    crop_special_prompt_file_name = os.getenv("CROP_SPECIAL_PROMPT_FILE_NAME")
    crops_file_name = os.getenv("CROPS_FILE_NAME")
    authors_file_name = os.getenv("AUTHORS_FILE_NAME")
    related_elements_file_name = os.getenv("RELATED_ELEMENTS_FILE_NAME")
    crop_special_results_directory_name = os.getenv("CROP_SPECIAL_RESULTS_DIRECTORY_NAME")

def load_prompt():
    with open(crop_special_prompt_file_name, "r", encoding="utf-8") as file:
        return file.read()

def load_elements(file_name, elements):
    with open(file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()

            if not line:
                break

            elements.append(line)

def load_crops():
    with open(crops_file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()
            
            if not line:
                break
            
            kor, eng = line.split()
            crops.append(Item(kor, eng))

def load_related_elements():
    with oepn(related_elements_file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readlind().strip()

            if not line:
                break
            
            kor, env = line.split()

            description = ""

            while True:
                line = file.readline().strip()

                if not line:
                    break
                
                description += line + "\n"
            
            related_elements.append(RelatedElement(kor, eng, description))

def create_crop_special_article():
    crop_special_human_message = create_human_messages(crop_special_article_prompt_template, crop, polarity, related_element)
    crop_special_article_result = llm.invoke(crop_special_human_message)

    return crop_special_article_result.content

def get_crop_special_article(result):
    crop_special_json_parser = JsonOutputParser(pydantic_object = CropSpecialArticle)

    return crop_special_json_parser.parse(result)

def create_article(crop_special_article):
    return Article(
        crop=crop.kor,
        title=crop_special_article["title"],
        body=crop_special_article["body"],
        author=random.choice(authors),
        change_rate=crop_special_article["change_rate"]
    )

def save_articles():
    saved_directory = f"{crop_special_results_directory_name}"
    os.makedirs(saved_directory, exist_ok=True)

    saved_file_name = f"{saved_directory}/{crop.eng}.txt"

    articles_dict = [article.dict() for article in articles]

    with open(saved_file_name, "w", encoding="utf-8") as file:
        json_string = json.dumps(articles_dict, ensure_ascii=False, indent=4)
        file.write(json_string)

def is_existed_articles():
    file_name = f"{crop_special_results_directory_name}/{crop.eng}.txt"

    return os.path.exists(file_name)


if __name__ == "__main__":
    set_env()

    crop_special_article_prompt_template = load_prompt()

    authors = []
    load_elements(authors_file_name, authors)

    crops = []
    load_crops()

    related_elements = []
    load_elements(related_elements_file_name, related_elements)

    polarities = ["가격이 증가하는", "가격이 감소하는"]

    # create client
    llm = ChatOpenAI(
        temperature = 1,
        model_name = model_name
    )

    for crop in crops:
        if is_existed_articles():
            print(f"pass {crop.eng}")
            continue

        print(f"start {crop.eng}")
        articles = []
        for related_element in related_elements:
            for polarity in polarities:
                try:
                    crop_special_article_result = create_crop_special_article()
                    crop_special_article = get_crop_special_article(crop_special_article_result)

                    articles.append(create_article(crop_special_article))
                except Exception as e:
                    print("An error occurred during article generation, retrying!")
                    print(e)
            
            print(f"{crop.eng}: {related_element}")
        
        save_articles()

        print(F"success saving {crop.eng}.txt")

        is_continued = True
        while True:
            user_input = input("Do you want to countinue? (y/n)").lower()
        
            if user_input == 'y':
                print("continue...")
                break
            elif user_input == 'n':
                is_continued = False
                print("Exiting the program...")
                break
            else:
                print("Invalid input. Please enter y or n")

        if not is_continued:
            break