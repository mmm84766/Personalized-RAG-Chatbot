from enum import Enum
from typing import Dict, List
from pydantic import BaseModel

class Tone(str, Enum):
    FORMAL = "formal"
    FRIENDLY = "friendly"
    HUMOROUS = "humorous"

class CommunicationGoal(str, Enum):
    EDUCATE = "educate"
    SUMMARIZE = "summarize"
    ADVISE = "advise"
    ENTERTAIN = "entertain"

class ResponseLength(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"

class ResponseStyle(str, Enum):
    STORYTELLING = "storytelling"
    BULLET_POINTS = "bullet points"
    STEP_BY_STEP = "step-by-step"

class UserPersona(str, Enum):
    BEGINNER = "beginner"
    DOMAIN_EXPERT = "domain expert"
    TEN_YEAR_OLD = "10-year-old"

class PersonalizationConfig(BaseModel):
    tone: Tone
    communication_goal: CommunicationGoal
    response_length: ResponseLength
    response_style: ResponseStyle
    language: str
    user_persona: UserPersona

# Default configuration
DEFAULT_CONFIG = PersonalizationConfig(
    tone=Tone.FRIENDLY,
    communication_goal=CommunicationGoal.EDUCATE,
    response_length=ResponseLength.DETAILED,
    response_style=ResponseStyle.STORYTELLING,
    language="English",
    user_persona=UserPersona.BEGINNER
)

# Sample persona configurations
PERSONA_CONFIGS: Dict[str, PersonalizationConfig] = {
    "Beginner": PersonalizationConfig(
        tone=Tone.FRIENDLY,
        communication_goal=CommunicationGoal.EDUCATE,
        response_length=ResponseLength.DETAILED,
        response_style=ResponseStyle.STEP_BY_STEP,
        language="English",
        user_persona=UserPersona.BEGINNER
    ),
    "Expert": PersonalizationConfig(
        tone=Tone.FORMAL,
        communication_goal=CommunicationGoal.ADVISE,
        response_length=ResponseLength.DETAILED,
        response_style=ResponseStyle.BULLET_POINTS,
        language="English",
        user_persona=UserPersona.DOMAIN_EXPERT
    ),
    "Young Learner": PersonalizationConfig(
        tone=Tone.HUMOROUS,
        communication_goal=CommunicationGoal.ENTERTAIN,
        response_length=ResponseLength.SHORT,
        response_style=ResponseStyle.STORYTELLING,
        language="English",
        user_persona=UserPersona.TEN_YEAR_OLD
    )
}

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "document_chunks" 