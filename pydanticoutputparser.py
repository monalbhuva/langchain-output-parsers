from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt = 18, description='Age of the person')
    city : str = Field(description='Name of the city the person belongs to')


parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({'place' : 'indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser

final_result = chain.invoke({'place' : 'Dubai'})
print(final_result)