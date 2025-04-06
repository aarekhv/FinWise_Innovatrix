from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

# Streamlit_Interface
st.set_page_config("FinWise", page_icon="📈")
st.sidebar.header("What is FinWise?")
st.sidebar.markdown("FinWise is a chatbot trained to provide financial insights and stock analysis.")
st.sidebar.header("Chat History:")
st.title("📈FinWise!")
st.caption("🚀 A chatbot trained for you to get financial insights and stock analysis.")
with st.chat_message("assistant"):
    st.markdown("Hello! I am FinWise, your financial assistant. How may I help you?")
userinput = st.chat_input()


# Web Search Agent
web_search_agent = Agent(
    name='Web Search AI Agent',
    role="Search the web for information",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
finance_agent = Agent(
    name='Financial AI Agent',
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[
        YFinanceTools(stock_price=True, company_news=True,analyst_recommendations=True, stock_fundamentals=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Combined Agent
multi_ai_agent = Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.2-90b-vision-preview"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Initializing MultiAIAgent
if userinput:
    with st.chat_message("user"):
        st.markdown(userinput)

    with st.chat_message("assistant"):
        response = multi_ai_agent.run(userinput)


        for message in response.messages[::-1]:
            if message.role == "assistant" and message.content:
                final_message = message.content
                break

        if final_message:
            st.markdown(final_message)
        else:
            st.markdown("⚠️ Sorry, I couldn't generate a response.")
