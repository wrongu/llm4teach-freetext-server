from ..config import OpenAIConfig
from ..feedback_providers.FeedbackProvider import FeedbackProvider
from ..prompt_stores import PromptStore
from ..llm4text_types import Assignment, Feedback, Submission
from langchain.prompts import ChatPromptTemplate
from langchain.llms.openai import OpenAIChat
from langchain.chains import LLMChain
from typing import List, Optional
import json


class LangchainFeedbackProvider(FeedbackProvider):
    """A feedback provider that transacts directly with the OpenAI API for student responses."""

    def __init__(
        self, prompt_store: PromptStore, config_override: Optional[OpenAIConfig] = None
    ):
        self.prompt_store = prompt_store
        if config_override is not None:
            self.config = config_override
        else:
            self.config = OpenAIConfig()
        self.llm = OpenAIChat(
            openai_api_key=self.config.token,
            model_name=self.config.model,
            openai_organization=self.config.organization,
        )

    async def get_feedback(
        self, submission: Submission, assignment: Assignment
    ) -> list[Feedback]:
        """
        Returns the feedback.
        Arguments:
            submission: The submission to provide feedback for.
            assignment: The assignment to provide feedback for.
        Returns:
            A list of feedback objects.
        """
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_store.get_prompt("langchain.system")),
                ("user", self.prompt_store.get_prompt("langchain.feedback")),
            ]
        )
        chain = LLMChain(llm=self.llm, template=template)
        output = chain.run(
            course_level="master's level",
            course_topic="computer vision",
            question=assignment.student_prompt,
            criteria="\n".join(
                [f"     * {f}" for f in assignment.feedback_requirements]
            ),
            response=submission.submission_string,
        )
        try:
            feedback = json.loads(output[-1]["feedback to student"])
            return [
                Feedback(
                    feedback_string=feedback,
                    source="LangchainFeedbackProvider[" + str(self.llm) + "]",
                    location=(0, len(submission.submission_string)),
                )
            ]
        except Exception as e:
            print(e)
            return []

    async def suggest_criteria(self, assignment: Assignment) -> List[str]:
        raise NotImplementedError()

    async def suggest_question(self, assignment: Assignment) -> str:
        raise NotImplementedError()


__all__ = [
    "LangchainFeedbackProvider",
]
