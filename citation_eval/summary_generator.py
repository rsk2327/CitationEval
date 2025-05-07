import asyncio

from autogen_agentchat.agents import AssistantAgent

from autogen_core.models import ChatCompletionClient


class SummaryGenerator:
    """
    A class to generate summaries for PDF documents using AutoGen.
    Supports both async and sync usage.
    """

    def __init__(
            self, model_client: ChatCompletionClient,
            temperature: float = 0.3,
            max_document_size:int = 15000
    ):
        """
        Initialize the summary generator.

        Args:
            model_name: The model to use for summarization
            temperature: The temperature for the model (0.0-1.0)
        """


        self.model_client = model_client
        self.temperature = temperature
        self.max_document_size = max_document_size
        self._agent = None


    def _create_agent(self) -> AssistantAgent:
        """Create the summarizer agent."""
        if self._agent is None:
            self._agent = AssistantAgent(
                name="document_summarizer",
                system_message="""You are a concise document summarizer. 
                Provide 1-2 line summaries that capture the main topic or purpose of documents.
                Each summary should be complete but brief, focusing on the core content.
                Keep summaries under 150 characters when possible.""",
                model_client=self.model_client
            )
        return self._agent

    async def generate_async(self, text: str) -> str:
        """
        Generate a summary for the given text (async version).

        Args:
            text: The text to summarize

        Returns:
            A 1-2 line summary of the text
        """
        # Truncate if text is too long
        if len(text) > self.max_document_size:
            text = text[:self.max_document_size] + "\n... [document truncated]"

        # Create the agent
        agent = self._create_agent()

        # Create the prompt
        prompt = f"""Please provide a 1-2 line summary of this document. 
        Focus on the main topic, purpose, or key finding.

        Document:
        {text}

        Summary:"""

        # Get the summary
        response = await agent.run(task=prompt)

        summary = response.messages[1].content.strip()

        return summary

    def generate(self, text: str) -> str:
        """
        Generate a summary for the given text (sync version).

        Args:
            text: The text to summarize

        Returns:
            A 1-2 line summary of the text
        """
        # Run the async function in a sync context
        return asyncio.run(self.generate_async(text))

    def close(self):
        """Close the model client to free resources."""
        if self.model_client:
            asyncio.run(self.model_client.close())
            self.model_client = None
            self._agent = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


