from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage


chat_model = ChatOllama(
    model="gemma:7b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

messages = [HumanMessage(content="日本で最も高い山は？")]
chat_model(messages)

