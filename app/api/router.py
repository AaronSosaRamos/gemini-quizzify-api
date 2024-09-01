from app.api.error_utilities import LoaderError, ToolExecutorError
from app.api.features.document_loaders import get_docs
from app.api.features.quizzify import QuizBuilder
from app.api.schemas.schemas import QuizzifyArgs
from fastapi import APIRouter, Depends
from app.api.logger import setup_logger
from app.api.auth.auth import key_check

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/generate-quizzes")
async def submit_tool( data: QuizzifyArgs, _ = Depends(key_check)):
    try:
        logger.info(f"File URL loaded: {data.file_url}")

        docs = get_docs(data.file_url, data.file_type, True)
    
        output = QuizBuilder(question_type=data.question_type, topic=data.topic, lang=data.lang, verbose=True).create_questions(
            docs, data.n_questions)
    
    except LoaderError as e:
        error_message = e
        logger.error(f"Error in RAGPipeline -> {error_message}")
        raise ToolExecutorError(error_message)
    
    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return output