from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# こちらを参考にしたもの： https://qiita.com/wwwcojp/items/c7f43c5f964b8db8a890#agents
# 実行には SERPAPI_API_KEY を環境変数二セットしておく必要あり。

llm = ChatOllama(
    model="gemma:7b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

llm = ChatOllama(
    model="gemma:7b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("日本で一番高い山の高さは何メートルですか?その高さの5乗根を求めてください。")
