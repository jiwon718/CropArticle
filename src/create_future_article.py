from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, validator
from typing import List

class SubCrop(BaseModel):
    name: str
    change_rate: float

    @validator("name")
    def validate_name(cls, value):
        if not value or value.strip() == "":
            raise ValueError("sub crop name not be empty")
        
        return value
    
    @validator("change_rate")
    def validate_change_rate(cls, value):
        if value is None:
            raise ValueError("sub crop change rate not be empty")
        elif abs(value) < 0:
            value = value * 100
        
        return value

class FutureArticle(BaseModel):
    title: str
    body: str
    change_rate: float
    spawn_rate: float
    sub_crops: List[SubCrop]

    @validator("title")
    def validate_title(cls, value):
        if not value or value.strip() == "":
            raise ValueError("future article title not be empty")
        
        return value
    
    @validator("body")
    def validate_body(cls, value):
        if not value or value.strip() == "":
            raise ValueError("future article body not be empty")
        
        return value
    
    @validator("change_rate")
    def validate_change_rate(cls, value):
        if value is None:
            raise ValueError("future article change rate not be empty")
        elif abs(value) < 0:
            print("future article change rate")
            value = value * 100
        
        return value
    
    @validator("spawn_rate")
    def validate_spawn_rate(cls, value):
        if value is None:
            raise ValueError("future article spawn rate not be empty")
        elif abs(value) < 0:
            value = value * 100
        
        return value

class FutureArticles(BaseModel):
    future_articles: List[FutureArticle]

# create prompt template
def create_prompt(template):
    prompt = PromptTemplate(
        template = template,
        input_variables=[
            "crop"
            "article_body"
        ]
    )

    return prompt

# create parser
def create_parser():
    return PydanticOutputParser(pydantic_object=FutureArticles)

# create output parser & human messages
def create_human_messages(template, crop, polarity, article_body):
    prompt = create_prompt(template)
    parser = create_parser()

    human_messages = [
        HumanMessage(content=prompt.format(crop=crop, polarity=polarity, article_body=article_body)),
        HumanMessage(content=parser.get_format_instructions())
    ]

    return human_messages
