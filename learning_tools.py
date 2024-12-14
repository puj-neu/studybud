from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate

class LessonPlanGenerator:
    def __init__(self, api_key: str):
        self.chat = ChatOpenAI(temperature=0.0, api_key=api_key)
        response_schemas = [
            ResponseSchema(name="week_plan", description="Daily learning objectives and activities for 7 days"),
            ResponseSchema(name="topics", description="Main topics to be covered"),
            ResponseSchema(name="resources", description="Additional learning resources and tips")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.prompt = ChatPromptTemplate.from_template(
            """Create a 7-day lesson plan from this content: {content}
            Break down the material into daily learning objectives.
            {format_instructions}
            """
        )
    
    def generate_plan(self, content: str) -> dict:
        messages = self.prompt.format_messages(
            content=content,
            format_instructions=self.format_instructions
        )
        response = self.chat.invoke(messages)
        return self.output_parser.parse(response.content)


class QuizGenerator:
    def __init__(self, api_key: str):
        self.chat = ChatOpenAI(temperature=0.0, api_key=api_key)
        
        response_schemas = [
            ResponseSchema(name="questions", description="List of quiz questions"),
            ResponseSchema(name="answers", description="Corresponding answers with explanations"),
            ResponseSchema(name="difficulty", description="Difficulty level of each question")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        
        template = """Generate a quiz based on this content: {content}
        Create varied question types (multiple choice, short answer, etc.)
        {format_instructions}
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)

    def generate_quiz(self, content: str) -> dict:
        messages = self.prompt.format_messages(
            content=content,
            format_instructions=self.format_instructions
        )
        response = self.chat.invoke(messages)
        return self.output_parser.parse(response.content)

