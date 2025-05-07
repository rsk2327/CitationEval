
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from citation_eval.agents import get_agents
import os

async def main():
    claude_llm = AnthropicChatCompletionClient(
        model="claude-3-opus-20240229",
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )

    researcher_agent, writer_agent, code_agent, search_agent = get_agents(claude_llm)

    text_mention_termination = TextMentionTermination("RESEARCH COMPLETE")

    selector_prompt = """Select an agent to perform task from the list of agents below

                          {roles} \
                          Current conversation context:
                          {history} \
                          Using the conversation context to determine the next agent to call using the following rules:
                          - If the last message was \
                      from ResearchAgent, then go to the Agent selected by the ResearchAgent
                          - If the last message is \
                      from any other agent, go to ResearchAgent who will decide which agent to go to next \
        
        Your output should be just the name of the Agent and nothing else. Do not include any explanation or any other text.

                      """

    group_chat = SelectorGroupChat(
        [researcher_agent, search_agent, code_agent, writer_agent],
        selector_prompt=selector_prompt,
        termination_condition=text_mention_termination,
        model_client=claude_llm,
        max_turns=3
    )

    task = """
    I want to learn how weather patterns affect solar panel efficiency.
    """

    result = await (group_chat.run(task=task))

    claude_llm.close()
    return result


if __name__ == '__main__':
    result = asyncio.run(main())
    a = 1
    print("**"*15)
    print("**" * 15)
    # print(result)
    for message in result.messages:
        print(message)





