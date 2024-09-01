from typing import List, Dict
import os

from app.api.logger import setup_logger
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv, find_dotenv

from app.api.schemas.schemas import QuizzifyArgs

load_dotenv(find_dotenv())

relative_path = "features/quzzify"

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class QuizBuilder:
    def __init__(self, topic, question_type, lang='en', vectorstore_class=Chroma, prompt=None, 
                 embedding_model=None, model=None, parser=None, verbose=False):
        self.question_type = question_type
        
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.5-pro"),
            "embedding_model": GoogleGenerativeAIEmbeddings(model='models/embedding-001'),
            "parser": self.get_parser_for_question_type(),
            "prompt": read_text_file("prompt/quizzify-prompt.txt"),
            "vectorstore_class": Chroma
        }
        
        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        
        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.vectorstore, self.retriever, self.runner = None, None, None
        self.topic = topic
        self.lang = lang
        self.verbose = verbose
        
        if vectorstore_class is None: raise ValueError("Vectorstore must be provided")
        if topic is None: raise ValueError("Topic must be provided")

    def get_parser_for_question_type(self):
        schema_mapping = {
            'fill_in_the_blank': FillInTheBlankQuestion,
            'open_ended': OpenEndedQuestion,
            'true_false': TrueFalseQuestion,
            'multiple_choice': MultipleChoiceQuestion,
            'relate_concepts': RelateConceptsQuestion,
            'math_exercises': MathExerciseQuestion,
            'default': MultipleChoiceQuestion,

        }
        schema = schema_mapping.get(self.question_type)
        if schema is None:
            raise ValueError(f"Unsupported question type: {self.question_type}")
        return JsonOutputParser(pydantic_object=schema)

    
    def compile(self, documents: List[Document]):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["attribute_collection"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        if self.runner is None:
            logger.info(f"Creating vectorstore from {len(documents)} documents") if self.verbose else None
            self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)
            logger.info(f"Vectorstore created") if self.verbose else None

            self.retriever = self.vectorstore.as_retriever()
            logger.info(f"Retriever created successfully") if self.verbose else None

            self.runner = RunnableParallel(
                {"context": self.retriever, 
                "attribute_collection": RunnablePassthrough()
                }
            )
        
        chain = self.runner | prompt | self.model | self.parser
        
        if self.verbose: logger.info(f"Chain compilation complete")
        
        return chain
    
    def validate_question(self, result):
        try:
            logger.info(f"Validating question format") if self.verbose else None
            schema = self.get_parser_for_question_type().pydantic_object
            schema(**result)
            return True
        except Exception as e:
            logger.warning(f"Invalid question format: {e}") if self.verbose else None
            return False
    
    def create_questions(self, documents: List[Document], num_questions: int = 5) -> List[Dict]:
        if self.verbose: logger.info(f"Creating {num_questions} questions")
        
        if num_questions > 10:
            return {"message": "error", "data": "Number of questions cannot exceed 10"}
        
        chain = self.compile(documents)
        
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 5  # Allow for more attempts to generate questions

        while len(generated_questions) < num_questions and attempts < max_attempts:
            response = chain.invoke(f"Topic: {self.topic}, Lang: {self.lang}, Question type: {self.question_type}")
            if self.verbose:
                logger.info(f"Generated response attempt {attempts + 1}: {response}")

            if "model_config" in response:
                del response["model_config"]

            if self.validate_question(response):
                generated_questions.append(response)
                if self.verbose:
                    logger.info(f"Valid question added: {response}")
                    logger.info(f"Total generated questions: {len(generated_questions)}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            
            attempts += 1

        if len(generated_questions) < num_questions:
            logger.warning(f"Only generated {len(generated_questions)} out of {num_questions} requested questions")
        
        if self.verbose: logger.info(f"Deleting vectorstore")
        self.vectorstore.delete_collection()
        
        return generated_questions[:num_questions]
    
#Fill-in-the-blank question type
class QuestionBlank(BaseModel):
    key: str = Field(description="A unique identifier for the blank, starting from 0.")
    value: str = Field(description="The text content to fill in the blank")

class FillInTheBlankQuestion(BaseModel):
    question: str = Field(description="The question text with blanks indicated by placeholders (It must be 5 blank spaces {0}, {1}, {2}, {3}, {4})")
    blanks: List[QuestionBlank] = Field(description="A list of blanks for the question, each with a key and a value")
    word_bank: List[str] = Field(description="A list of the correct texts that fill in the blanks, in random order")
    explanation: str = Field(description="An explanation of why the answers are correct")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "The {0} of France is {1}, and it is known for its {2} and {3} {4}.",
                "blanks": [
                    {"key": "0", "value": "capital"},
                    {"key": "1", "value": "Paris"},
                    {"key": "2", "value": "art"},
                    {"key": "3", "value": "culinary"},
                    {"key": "4", "value": "delights"}
                ],
                "word_bank": ["delights", "art", "Paris", "culinary", "capital"],
                "explanation": "Paris is the capital of France, and it is renowned for its contributions to art and its exceptional culinary scene."
              }
          """
        }
    }

#Open-ended question type
class OpenEndedQuestion(BaseModel):
    question: str = Field(description="The open-ended question text")
    answer: str = Field(description="The expected correct answer")
    feedback: List[str] = Field(description="A list of possible answers for the provided question")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "What is the significance of Paris in French history?",
                "answer": "Paris is the capital of France and has been a major center for politics, culture, art, and history.",
                "feedback": [
                    "Paris is the capital of France.",
                    "Paris has been a cultural center in Europe.",
                    "Paris played a major role in the French Revolution."
                ]
              }
          """
        }
    }

#True-False question type
class TrueFalseQuestion(BaseModel):
    question: str = Field(description="The True-False question text")
    answer: bool = Field(description="The correct answer, either True or False")
    explanation: str = Field(description="An explanation of why the answer is correct")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "The Eiffel Tower is located in Paris.",
                "answer": true,
                "explanation": "The Eiffel Tower is a famous landmark located in Paris, France."
              }
          """
        }
    }

#Multiple Choice question type
class QuestionChoice(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, or D.")
    value: str = Field(description="The text content of the choice")
class MultipleChoiceQuestion(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoice] = Field(description="A list of choices for the question, each with a key and a value")
    answer: str = Field(description="The key of the correct answer from the choices list")
    explanation: str = Field(description="An explanation of why the answer is correct")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "What is the capital of France?",
                "choices": [
                    {"key": "A", "value": "Berlin"},
                    {"key": "B", "value": "Madrid"},
                    {"key": "C", "value": "Paris"},
                    {"key": "D", "value": "Rome"}
                ],
                "answer": "C",
                "explanation": "Paris is the capital of France."
              }
          """
        }

      }

#Relate concepts question type
class TermMeaningPair(BaseModel):
    term: str = Field(description="The term to be matched")
    meaning: str = Field(description="The meaning of the term")

class RelateConceptsQuestion(BaseModel):
    question: str = Field(description="The 'Relate concepts' question text. It must be appropriate for generating pairs and answers.")
    pairs: List[TermMeaningPair] = Field(description="A list of term-meaning pairs in disorder")
    answer: List[TermMeaningPair] = Field(description="A list of the correct term-meaning pairs in order")
    explanation: str = Field(description="An explanation of the correct term-meaning pairs")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "Match each term with its correct meaning.",
                "pairs": [
                    {
                        "term": "Chlorophyll",
                        "meaning": "The powerhouse of the cell, where respiration and energy production occur."
                    },
                    {
                        "term": "Photosynthesis",
                        "meaning": "A green pigment responsible for the absorption of light to provide energy for photosynthesis."
                    },
                    {
                        "term": "Mitochondria",
                        "meaning": "The process by which green plants use sunlight to synthesize foods with the help of chlorophyll."
                    },
                    {
                        "term": "Nucleus",
                        "meaning": "The gel-like substance inside the cell membrane."
                    },
                    {
                        "term": "Cytoplasm",
                        "meaning": "The control center of the cell that contains DNA."
                    }
                ],
                "answer": [
                    {
                        "term": "Photosynthesis",
                        "meaning": "The process by which green plants use sunlight to synthesize foods with the help of chlorophyll."
                    },
                    {
                        "term": "Chlorophyll",
                        "meaning": "A green pigment responsible for the absorption of light to provide energy for photosynthesis."
                    },
                    {
                        "term": "Mitochondria",
                        "meaning": "The powerhouse of the cell, where respiration and energy production occur."
                    },
                    {
                        "term": "Nucleus",
                        "meaning": "The control center of the cell that contains DNA."
                    },
                    {
                        "term": "Cytoplasm",
                        "meaning": "The gel-like substance inside the cell membrane."
                    }
                ],
                "explanation": "Photosynthesis involves using sunlight to create food in plants, facilitated by chlorophyll. Mitochondria are involved in energy production in cells. The nucleus is the control center of the cell, and the cytoplasm is the gel-like substance within the cell membrane."
              }
          """
        }
    }

#Math. Exercise question type
class MathExerciseQuestion(BaseModel):
    question: str = Field(description="The math exercise question text")
    solution: str = Field(description="The step-by-step solution to the math problem")
    correct_answer: str = Field(description="The correct answer to the math problem")
    explanation: str = Field(description="An explanation of why the solution is correct")

    model_config = {
        "json_schema_extra": {
            "examples": """ 
                {
                "question": "Solve the equation: 2x + 3 = 11",
                "solution": "Step 1: Subtract 3 from both sides to get 2x = 8. Step 2: Divide both sides by 2 to get x = 4.",
                "correct_answer": "4",
                "explanation": "By isolating the variable x, we find that x equals 4."
              }
          """
        }
    }