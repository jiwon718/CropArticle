from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class CropArticle(BaseModel):
    title: str = Field(description="재목")
    body: str = Field(description="본문")

    @validator("title")
    def validate_title(cls, field):
        if not field or field.strip() == "":
            raise ValueError("crop article title must not be empty")
        
        return field
    
    @validator("body")
    def validate_body(cls, field):
        if not field or field.strip() == "":
            raise ValueError("crop article body must not be empty")
        
        return field

# create prompt template
def create_prompt(template):
    prompt = PromptTemplate(
        template = template,
        input_variables=[
            "crop",
            "aspect"
        ]
    )

    return prompt

# create output parser & human messages
def create_human_messages(template, crop, aspect, polarity):
    prompt = create_prompt(template)
    parser = PydanticOutputParser(pydantic_object=CropArticle)

    human_messages = [
        HumanMessage(content=prompt.format(crop=crop, aspect=aspect, polarity=polarity)),
        HumanMessage(content=parser.get_format_instructions())
    ]

    return human_messages