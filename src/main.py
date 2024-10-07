import random
import json
import time
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel
from typing import List

from create_crop_article import CropArticle, create_human_messages as create_crop_human_messages
from create_future_article import SubCrop, FutureArticles, create_human_messages as create_future_human_messages


class Item():
    def __init__(self, kor, eng):
        self.kor = kor
        self.eng = eng

class Aspect(Item):
    def __init__(self, kor, eng, description):
        super().__init__(kor, eng)
        self.description = description

class SubCropWithId(BaseModel):
    id: str
    name: str
    change_rate: float

class FutureArticleWithAuthor(BaseModel):
    title: str
    body: str
    author: str
    change_rate: float
    spawn_rate: float
    sub_crops: List[SubCropWithId]

class Article(BaseModel):
    crop: str
    aspect: str
    title: str
    body: str
    author: str
    future_articles: List[FutureArticleWithAuthor]


def set_env():
    global model_name, article_count
    global crop_prompt_file_name, future_prompt_file_name, retried_crop_prompt_file_name, retried_future_prompt_file_name, aspects_file_name, crops_file_name, authors_file_name, results_directory_name
    global generation_time, stop_time

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    model_name = os.getenv("MODEL_NAME")
    article_count = int(os.getenv("ARTICLE_COUNT"))
    crop_prompt_file_name = os.getenv("CROP_PROMPT_FILE_NAME")
    future_prompt_file_name = os.getenv("FUTURE_PROMPT_FILE_NAME")
    retried_crop_prompt_file_name = os.getenv("RETRIED_CROP_PROMPT_FILE_NAME")
    retried_future_prompt_file_name = os.getenv("RETRIED_FUTURE_PROMPT_FILE_NAME")
    aspects_file_name = os.getenv("ASPECTS_FILE_NAME")
    crops_file_name = os.getenv("CROPS_FILE_NAME")
    authors_file_name = os.getenv("AUTHORS_FILE_NAME")
    results_directory_name = os.getenv("RESULTS_DIRECTORY_NAME")
    generation_time = int(os.getenv("GENERATION_TIME"))
    stop_time = int(os.getenv("STOP_TIME"))

def load_prompt(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()

def load_authors():
    with open(authors_file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()

            if not line:
                break

            authors.append(line)

def load_crops():
    with open(crops_file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()
            
            if not line:
                break
            
            kor, eng = line.split()
            crops.append(Item(kor, eng))

def load_aspects():
    with open(aspects_file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()

            if not line:
                break
            
            kor, eng = line.split()

            description = ""
            while True:
                line = file.readline().strip()

                if not line:
                    break
                
                description += line + "\n"
            
            aspects.append(Aspect(kor, eng, description))


def create_crop_article_id():
    return crop.eng + "_" + aspect.eng

def create_crop_article():
    crop_human_messages = create_crop_human_messages(crop_article_prompt_template, crop, aspect, polarity)
    crop_article_result = llm.invoke(crop_human_messages)

    return crop_article_result.content

def create_future_articles(crop_article):
    future_human_messages = create_future_human_messages(future_article_prompt_template, crop, future_polarity, crop_article["body"])
    future_articles_result = llm.invoke(future_human_messages)

    return future_articles_result.content

def get_crop_article(result):
    crop_json_parser = JsonOutputParser(pydantic_object=CropArticle)

    return crop_json_parser.parse(result)

def get_future_articles(result):
    future_json_parser = JsonOutputParser(pydantic_object=FutureArticles)
    
    return future_json_parser.parse(result)

def create_article(crop_article, future_articles):
    future_articles_with_author = []
    for future_article in future_articles["future_articles"]:
        # sub crop
        sub_crops = []
        for sub_crop in future_article["sub_crops"]:
            name = sub_crop["name"]
            # validate and translate crop
            english_name = translate_english(name)

            if english_name:
                sub_crop_with_id = SubCropWithId(
                    id=english_name,
                    name=name,
                    change_rate=sub_crop["change_rate"]
                )
                sub_crops.append(sub_crop_with_id)

        future_article_with_author = FutureArticleWithAuthor(
            title=future_article["title"],
            body=future_article["body"],
            author=random.choice(authors),
            change_rate=future_article["change_rate"],
            spawn_rate=future_article["spawn_rate"],
            sub_crops=sub_crops
        )
        future_articles_with_author.append(future_article_with_author)

    return Article(
        crop=crop.kor,
        aspect=aspect.kor,
        title=crop_article["title"],
        body=crop_article["body"],
        author=random.choice(authors),
        future_articles=future_articles_with_author
    )

def translate_english(crop_kor):
    for crop in crops:
        if crop_kor == crop.kor:
            return crop.eng
    
    return None

def save_articles(num):
    saved_directory = f"{results_directory_name}/{crop.eng}"
    os.makedirs(saved_directory, exist_ok=True)

    saved_file_name = f"{saved_directory}/{aspect.eng}_{num}.txt"

    articles_dict = [article.dict() for article in articles]

    with open(saved_file_name, "w", encoding="utf-8") as file:
        json_string = json.dumps(articles_dict, ensure_ascii=False, indent=4)
        file.write(json_string)

def is_existed_articles(num):
    file_name = f"{results_directory_name}/{crop.eng}/{aspect.eng}_{num}.txt"

    return os.path.exists(file_name)


if __name__ == "__main__":
    set_env()

    crop_article_prompt_template = load_prompt(crop_prompt_file_name)
    future_article_prompt_template = load_prompt(future_prompt_file_name)

    authors = []
    load_authors()

    crops = []
    load_crops()

    aspects = []
    load_aspects()

    # create client
    llm = ChatOpenAI(
        temperature = 0.5,
        model_name = model_name
    )

    for crop in crops:
        for aspect in aspects:
            print(f"start {crop.eng}: {aspect.eng}")

            # initialize prompt message
            crop_article_prompt_template = load_prompt(crop_prompt_file_name)
            future_article_prompt_template = load_prompt(future_prompt_file_name)

            crop_article_id = create_crop_article_id()

            # create articles
            last_count = article_count // 10
            mid_count = article_count // 2
            for num in range(0, last_count):
                if is_existed_articles(num):
                    print(f"pass {crop.eng}: {aspect.eng} - {num}")
                    continue

                crop_articles = []
                articles = []

                start = num * 10 + 1
                end = num * 10 + 11
                # create crop article
                count = start
                while count < end:
                    polarity = "가격이 증가하는"
                    if count > mid_count:
                        polarity = "가격이 감소하는"

                    try:
                        crop_article_result = create_crop_article()
                        crop_article = get_crop_article(crop_article_result)

                        crop_articles.append(crop_article)
                        
                        count += 1
                    except Exception as e:
                        print("An error occurred during article generation, retrying!")
                        count -= 1

                print(f"success article about {crop.eng}")

                #create future article
                count = start
                while count < end:
                    crop_article = crop_articles[count - start]
                    future_polarity = "급격한 증가"
                    if count > mid_count:
                        future_polarity = "급격한 감소"

                    try:
                        future_articles_result = create_future_articles(crop_article)
                        future_articles = get_future_articles(future_articles_result)

                        articles.append(create_article(crop_article, future_articles))
                        
                        print(f"[{count}] {crop_article_id}: {crop_article["title"]}")

                        count += 1
                    except Exception as e:
                        print("An error occurred during article generation, retrying!")
                        count -= 1
                
                save_articles(num)

                print(f"success saving {crop.eng}/{aspect.eng}_{num}.txt")

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
