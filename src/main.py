import random
import json
import os

from dotenv import load_dotenv

from langchain.schema import HumanMessage
from langchain_community.llms import Ollama
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel
from langchain_core.pydantic_v1 import BaseModel
from langchain_redis import RedisChatMessageHistory
from typing import List

from create_crop_article import CropArticle, create_human_messages as create_crop_human_messages
from create_future_article import SubCrop, FutureArticles, create_human_messages as create_future_human_messages


class Item():
    def __init__(self, kor, eng):
        self.kor = kor
        self.eng = eng

class FutureArticleWithAuthor(BaseModel):
    title: str
    body: str
    author: str
    change_rate: float
    spawn_rate: float
    sub_crops: List[SubCrop]

class Article(BaseModel):
    crop: str
    title: str
    body: str
    author: str
    future_articles: List[FutureArticleWithAuthor]


def set_env():
    global model_name, article_count
    global crop_prompt_file_name, future_prompt_file_name, aspects_file_name, crops_file_name, authors_file_name, results_directory_name

    load_dotenv()

    model_name = os.getenv("MODEL_NAME")
    article_count = int(os.getenv("ARTICLE_COUNT"))
    crop_prompt_file_name = os.getenv("CROP_PROMPT_FILE_NAME")
    future_prompt_file_name = os.getenv("FUTURE_PROMPT_FILE_NAME")
    aspects_file_name = os.getenv("ASPECTS_FILE_NAME")
    crops_file_name = os.getenv("CROPS_FILE_NAME")
    authors_file_name = os.getenv("AUTHORS_FILE_NAME")
    results_directory_name = os.getenv("RESULTS_DIRECTORY_NAME")

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

def load_items(file_name, items):
    with open(file_name, "r", encoding="utf-8") as file:
        while True:
            line = file.readline().strip()
            
            if not line:
                break
            
            kor, eng = line.split()
            items.append(Item(kor, eng))

def create_redis_session_id():
    return crop.eng + "_" + aspect.eng

def get_redis_history():
    return RedisChatMessageHistory(
        session_id=redis_session_id,
        redis_url="redis://localhost:6379/0"
    )

def create_crop_future_articles():
    crop_article_result = create_crop_article()
    crop_article = get_crop_article(crop_article_result)

    future_articles_result = create_future_articles(crop_article)
    future_articles = get_future_articles(future_articles_result)

    return crop_article, future_articles

def create_crop_article():
    crop_human_messages = create_crop_human_messages(crop_article_prompt_template, crop.kor, aspect.kor)
    crop_article_result = chain_with_history.invoke(crop_human_messages)

    return crop_article_result["output_message"]

def create_future_articles(crop_article):
    future_human_messages = create_future_human_messages(future_article_prompt_template, crop.kor, crop_article["body"])
    future_articles_result = llm.invoke(future_human_messages)

    return future_articles_result

def get_crop_article(result):
    crop_json_parser = JsonOutputParser(pydantic_object=CropArticle)

    return crop_json_parser.parse(result)

def get_future_articles(result):
    future_json_parser = JsonOutputParser(pydantic_object=FutureArticles)
    
    return future_json_parser.parse(result)

def create_article():
    future_articles_with_author = []
    for future_article in future_articles["future_articles"]:
        future_article_with_author = FutureArticleWithAuthor(
            title=future_article["title"],
            body=future_article["body"],
            author=random.choice(authors),
            change_rate=future_article["change_rate"],
            spawn_rate=future_article["spawn_rate"],
            sub_crops=future_article["sub_crops"]
        )
        future_articles_with_author.append(future_article_with_author)

    return Article(
        crop=crop.kor,
        title=crop_article["title"],
        body=crop_article["body"],
        author=random.choice(authors),
        future_articles=future_articles_with_author
    )

def save_articles():
    saved_directory = f"{results_directory_name}/{crop.eng}"
    os.makedirs(saved_directory, exist_ok=True)

    saved_file_name = f"{saved_directory}/{aspect.eng}.txt"

    articles_dict = [article.dict() for article in articles]

    with open(saved_file_name, "w", encoding="utf-8") as file:
        json_string = json.dumps(articles_dict, ensure_ascii=False, indent=4)
        file.write(json_string)

def is_existed_articles():
    file_name = f"{results_directory_name}/{crop.eng}/{aspect.eng}.txt"

    return os.path.exists(file_name)


if __name__ == "__main__":
    set_env()

    crop_article_prompt_template = load_prompt(crop_prompt_file_name)
    future_article_prompt_template = load_prompt(future_prompt_file_name)

    authors = []
    load_authors()

    aspects = []
    load_items(aspects_file_name, aspects)

    crops = []
    load_items(crops_file_name, crops)

    # create client
    llm = Ollama(model=model_name)
    chain = RunnableParallel({"output_message": llm})

    for crop in crops:
        for aspect in aspects:
            if is_existed_articles():
                print(f"pass {crop.eng}: {aspect.eng}")
                continue

            print(f"start {crop.eng}: {aspect.eng}")

            redis_session_id = create_redis_session_id()

            chain_with_history = RunnableWithMessageHistory(
                chain,
                get_redis_history
            )

            articles = []

            # create articles
            for count in range(1, article_count + 1):
                crop_article, future_articles = create_crop_future_articles()

                articles.append(create_article())

                print(f"[{count}] {redis_session_id}: {crop_article["title"]}")
            
            save_articles()

            print(f"success saving {crop.eng}/{aspect.eng}.txt")
