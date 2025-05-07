import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from citation_eval.agents import get_agents
import os
from citation_eval.ingestion import PDFCollection
from citation_eval.summary_generator import SummaryGenerator
from citation_eval.toc_generator import TableOfContentsGenerator
from pathlib import Path

def main(question:str,
         answer:str,
         answer_part:str,
         citation:str,
         pdf_directory:Path,
         output_dir:Path,
         ):

    claude_llm = AnthropicChatCompletionClient(
        model="claude-3-opus-20240229",
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )

    # read pdf data
    collection = PDFCollection(pdf_directory=pdf_directory, cache_directory = output_dir)
    collection.process_pdfs(force_reprocess  = True)

    # generate summaries for each pdf
    summary_generator = SummaryGenerator(claude_llm)
    toc_generator = TableOfContentsGenerator(claude_llm)

    for filename in collection.documents:

        summary = summary_generator.generate(collection.documents[filename].full_text)
        collection.add_document_field(filename, 'doc_summary', summary)

        toc = toc_generator.generate(collection.documents[filename].full_text)
        collection.add_document_field(filename, 'toc', toc)



    a = 1


if __name__ == '__main__':

    pdf_directory = Path('/Users/roshansk/Documents/GitHub/CitationEval/data/pdfs')
    output_directory = Path('/Users/roshansk/Documents/temp/citation_eval_output')
    question = ''
    answer = ''
    answer_part = ''
    citation = ''

    main(
        question=question,
        answer=answer,
        answer_part=answer_part,
        citation=citation,
        pdf_directory=pdf_directory,
        output_dir=output_directory,
    )
