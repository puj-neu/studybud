from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models import Flashcard

class FlashcardGeneratorOpenAI:
    def __init__(self, api_key: str, llm_model: str = "gpt-3.5-turbo"):
        self.chat = ChatOpenAI(temperature=0.0, model=llm_model, api_key=api_key)
        
        response_schemas = [
            ResponseSchema(name="input_expression", description="The main concept or question"),
            ResponseSchema(name="output_expression", description="The explanation or answer"),
            ResponseSchema(name="example_usage", description="An example that illustrates the concept"),
            ResponseSchema(name="source", description="Reference to source material")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        
        template = """Generate a study flashcard based on the following content:
        Content: {content}
        
        Create a clear concept-explanation pair that helps understand the key idea.
        {format_instructions}
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)


    def generate_flashcard(self, content: str) -> Flashcard:
        messages = self.prompt.format_messages(
            content=content,
            format_instructions=self.format_instructions
        )
        response = self.chat.invoke(messages)
        flashcard_dict = self.output_parser.parse(response.content)
        return Flashcard(**flashcard_dict)
