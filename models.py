from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class Document:
    content: str
    metadata: dict

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatHistory:
    messages: List[ChatMessage]

    def append(self, user_message: str, assistant_message: str):
        self.messages.extend([
            ChatMessage(role="human", content=user_message),
            ChatMessage(role="assistant", content=assistant_message)
        ])

    def as_list(self):
        return [(msg.content, msg.content) for msg in self.messages[::2]]

    from dataclasses import dataclass, asdict

@dataclass
class Flashcard:
    input_expression: str
    output_expression: str
    example_usage: str
    source: str = ""
    @classmethod
    def from_dict(cls, data: dict) -> "Flashcard":
        return cls(
            input_expression=data.get("input_expression", ""),
            output_expression=data.get("output_expression", ""),
            example_usage=data.get("example_usage", ""),
            source=data.get("source", "")
        )
@dataclass
class Flashcards:
    data: list[Flashcard]

    def as_json(self):
        return {"flashcards": [asdict(card) for card in self.data]}

    @classmethod
    def import_from_json(cls, data: dict) -> "Flashcards":
        flashcard_objects = [Flashcard(**card) for card in data["flashcards"]]
        return cls(data=flashcard_objects)

    def __len__(self):
        return len(self.data)
@dataclass
class Quiz:
    questions: List[dict]
    
@dataclass
class LessonPlan:
    objectives: List[str]
    activities: List[dict]
    resources: List[str]

