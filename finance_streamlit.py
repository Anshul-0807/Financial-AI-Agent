
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

## Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

## Multi AI Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit Interface
def streamlit_interface():
    st.title("AI Financial and Web Insights")
    st.markdown("This application uses AI agents to provide financial insights and search the web for information.")

    st.sidebar.header("Options")
    agent_choice = st.sidebar.selectbox("Choose Agent", ["Financial Agent", "Web Search Agent"])
    query = st.text_area("Enter your query:")

    if st.button("Get Response"):
        if not query:
            st.warning("Please enter a query.")
        else:
            st.info(f"Querying the {agent_choice}...")
            try:
                if agent_choice == "Financial Agent":
                    response = finance_agent.run(message=query, stream=False)
                elif agent_choice == "Web Search Agent":
                    response = web_search_agent.run(message=query, stream=False)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    streamlit_interface()

