import json
import asyncio
from typing import Dict, List, Tuple, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient


class TableOfContentsGenerator:
    """
    A class to generate or extract table of contents from PDF documents using AutoGen.
    Supports both existing TOC extraction and new TOC generation.
    """

    def __init__(self,
                 model_client: ChatCompletionClient,
                 temperature: float = 0.3,
                 max_document_size: int = 50000,
                 max_chunk_size: int = 25000):
        """
        Initialize the table of contents generator.

        Args:
            model_name: The model to use for TOC generation
            temperature: The temperature for the model (0.0-1.0)
            max_document_size: Maximum size of the document to process
            max_chunk_size: Maximum size of each chunk for LLM processing
        """
        self.model_client = model_client
        self.temperature = temperature
        self.max_document_size = max_document_size
        self.max_chunk_size = max_chunk_size

        self._agent = None


    def _create_agent(self) -> AssistantAgent:
        """Create the TOC generator agent."""
        if self._agent is None:
            self._agent = AssistantAgent(
                name="toc_generator",
                system_message="""You are a table of contents expert. Your task is to either extract an existing table of contents from the given text or generate a sensible one based on the content.

                Your response MUST be a valid JSON object with exactly two keys:
                1. "toc": A list of strings representing the table of contents items
                2. "found_toc": A boolean indicating whether an existing TOC was found (true) or generated (false)

                If you find an existing table of contents:
                - Look for sections with phrases like "Contents", "Table of Contents", "Index", etc.
                - Extract only the main headings/topics (remove page numbers, dots, etc.)
                - Set "found_toc" to true

                If no table of contents exists:
                - Generate a logical table of contents based on the document structure
                - Create 3-8 main sections that represent the document's key topics
                - Set "found_toc" to false

                Example response format:
                {
                    "toc": ["Introduction", "Background", "Methodology", "Results", "Conclusion"],
                    "found_toc": false
                }""",
                model_client=self.model_client
            )
        return self._agent

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        words = text.split()

        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.max_chunk_size:
                current_chunk += (word + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _process_chunk(self, chunk: str, chunk_index: int) -> Dict:
        """
        Process a single chunk to extract or generate TOC.

        Args:
            chunk: Text chunk to process
            chunk_index: Index of the chunk

        Returns:
            Dictionary with 'toc' and 'found_toc' keys
        """
        agent = self._create_agent()

        prompt = f"""Please analyze this text chunk (part {chunk_index + 1}) and either extract an existing table of contents or generate a logical one based on the content.

        Text chunk:
        {chunk}

        Respond with a JSON object containing:
        1. "toc": list of table of contents items
        2. "found_toc": boolean indicating if TOC was found (true) or generated (false)"""

        try:
            response = await agent.run(task=prompt)

            response_ = response.messages[1].content
            a = 1

            # Try to parse the JSON response
            try:
                result = json.loads(response_)

                # Validate the structure
                if 'toc' in result and 'found_toc' in result:
                    return result
                else:
                    # Fallback if structure is invalid
                    return {
                        "toc": [],
                        "found_toc": False
                    }

            except json.JSONDecodeError:
                # If parsing fails, try to extract a list manually
                # This is a fallback for when the LLM doesn't return valid JSON
                return {
                    "toc": [],
                    "found_toc": False
                }

        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            return {
                "toc": [],
                "found_toc": False
            }

    async def generate_async(self, text: str) -> Tuple[List[str], bool]:
        """
        Generate or extract table of contents from text (async version).

        Args:
            text: The text to process

        Returns:
            Tuple of (table of contents list, found_existing_toc boolean)
        """
        # Limit document size
        if len(text) > self.max_document_size:
            text = text[:self.max_document_size]

        # Split into chunks
        chunks = self._chunk_text(text)

        # Process chunks in parallel
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(self._process_chunk(chunk, i))

        # Wait for all chunks to be processed
        results = await asyncio.gather(*tasks)

        # Process results
        found_tocs = []
        generated_tocs = []

        for result in results:
            if result["found_toc"]:
                found_tocs.extend(result["toc"])
            else:
                generated_tocs.extend(result["toc"])

        # Determine final TOC and status
        if found_tocs:
            # Remove duplicates while preserving order
            seen = set()
            final_toc = []
            for item in found_tocs:
                if item not in seen:
                    seen.add(item)
                    final_toc.append(item)
            return final_toc, True
        else:
            # Combine all generated TOCs and remove duplicates
            seen = set()
            final_toc = []
            for item in generated_tocs:
                if item not in seen:
                    seen.add(item)
                    final_toc.append(item)
            return final_toc, False

    def generate(self, text: str) -> Tuple[List[str], bool]:
        """
        Generate or extract table of contents from text (sync version).

        Args:
            text: The text to process

        Returns:
            Tuple of (table of contents list, found_existing_toc boolean)
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