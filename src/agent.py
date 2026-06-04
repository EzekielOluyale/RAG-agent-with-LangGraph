from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent

from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest
)

def build_agent(vector_store):

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        vertexai=True,
        temperature=0
    )

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:

        last_query = (request.state["messages"][-1].content)

        retrieved_docs = (
            vector_store.similarity_search(
                last_query,
                k=4
            )
        )

        docs_content = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        return (
            "You are a helpful assistant. "
            "Answer using the retrieved context. "
            "If the answer is not present, "
            "say you don't know.\n\n"
            f"{docs_content}"
        )

    return create_agent(
        model=model,
        tools=[],
        middleware=[prompt_with_context]
    )