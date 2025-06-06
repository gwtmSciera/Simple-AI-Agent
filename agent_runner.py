from langchain.agents import initialize_agent, AgentType
from langchain_ollama.llms import OllamaLLM
from tools import tools

llm = OllamaLLM(model="llama3.1")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def ask_agent(question: str) -> str:
    return agent.run(question)