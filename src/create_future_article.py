from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List

class SubCrop(BaseModel):
    name: str = Field(description="작물명")
    change_rate: float = Field(description="등락률")

    @validator("name")
    def validate_name(cls, field):
        if not field or field.strip() == "":
            raise ValueError("sub crop name not be empty")
        
        return field
    
    @validator("change_rate")
    def validate_change_rate(cls, field):
        if field is None:
            raise ValueError("sub crop change rate not be empty")
        elif abs(field) < 0:
            field = field * 100
        
        return field

class FutureArticle(BaseModel):
    title: str = Field(description="제목")
    body: str = Field(description="본문")
    change_rate: float = Field(description="등략률")
    spawn_rate: float = Field(description="등장률")
    sub_crops: List[SubCrop] = Field(description="관련 작물 목록")

    @validator("title")
    def validate_title(cls, field):
        if not field or field.strip() == "":
            raise ValueError("future article title not be empty")
        
        return field
    
    @validator("body")
    def validate_body(cls, field):
        if not field or field.strip() == "":
            raise ValueError("future article body not be empty")
        
        return field
    
    @validator("change_rate")
    def validate_change_rate(cls, field):
        if field is None:
            raise ValueError("future article change rate not be empty")
        elif abs(field) < 0:
            print("future article change rate")
            field = field * 100
        
        return field
    
    @validator("spawn_rate")
    def validate_spawn_rate(cls, field):
        if field is None:
            raise ValueError("future article spawn rate not be empty")
        elif abs(field) < 0:
            field = field * 100
        
        return field

class FutureArticles(BaseModel):
    future_articles: List[FutureArticle] = Field(description="결과 기사 목록")

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
