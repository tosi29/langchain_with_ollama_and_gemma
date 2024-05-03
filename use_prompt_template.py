from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate


template_highest_in_japan = PromptTemplate(
    input_variables=["thing"],
    template="日本で最も高い{thing}は？",
)

chat_model = ChatOllama(
    model="gemma:7b",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

messages = [HumanMessage(content=template_highest_in_japan.format(thing="山"))]
chat_model(messages)

