from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.tools import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent

import sys
import pandas as pd

DATA_PATH = "titanic.csv"

llm = Ollama(
    temperature=0, 
)

def main():
        
    agent = create_agent(DATA_PATH);

    interaction(agent)

    # agent.invoke("how many rows are there?")

def create_agent(data_path): 
    agent = create_csv_agent(
        llm=llm, 
        path=data_path,
        verbose=True,
        pandas_kwargs = {"sep":",", "encoding":'Latin-1'},
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent

def create_agent2(data_path): 
    df = pd.read_csv(data_path)
    agent = create_pandas_dataframe_agent(
        llm, 
        df,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent


def interaction(csv_agent):
    tools = load_tools([], llm=llm)

    csv_tool = Tool(
        name="CSV Agent",
        func=csv_agent.run,
        description="Useful for interacting with CSV data.",
    )
    tools.extend([csv_tool])
    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
      
    )


    chat_history = []
    while True:
        query = input('Prompt: ')
        if query == "exit" or query == "quit" or query == "q":
            print('Exiting')
            sys.exit()
        # result = qa_chain({'question': query, 'chat_history': chat_history})
        result = agent.invoke("how many rows are there?")
        print('Answer: ' + result['answer'] + '\n')
        chat_history.append((query, result['answer']))

if __name__ == "__main__":
    main()
