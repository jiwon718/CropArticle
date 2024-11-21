from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, validator

class CropSpecialArticle(BaseModel):
    title: str
    body: str
    change_rate: float

    @validator("title")
    def validate_title(cls, value):
        if not value or value.strip() == "":
            raise ValueError("crop special article title must not be empty")
        
        return value
    
    @validator("body")
    def validate_body(cls, value):
        if not value or value.strip() == "":
            raise ValueError("crop special article body must not be empty")
        
        return value
    
    @validator("change_rate")
    def validate_change_rate(cls, value):
        if value is None:
            raise ValueError("crop change rate not be empty")
        elif abs(value) < 0:
            value = value * 100
        
        return value

# create prompt template
def create_prompt(template):
    prompt = PromptTemplate(
        template = template,
        input_variables = [
            "crop",
            "polarity",
            "related_element"
        ]
    )

    return prompt

# create output parser & human messages
def create_human_messages(template, crop, polarity, related_element):
    prompt = create_prompt(template)
    parser = PydanticOutputParser(pydantic_object=CropSpecialArticle)

    human_messages = [
        HumanMessage(content=prompt.format(crop=crop, polarity=polarity, related_element=related_element)),
        HumanMessage(content=parser.get_format_instructions())
    ]

    return human_messages