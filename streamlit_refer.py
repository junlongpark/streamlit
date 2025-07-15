import streamlit as st
import wikipedia
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback

# ----- Streamlit 앱 시작 -----
def main():
    st.set_page_config(
        page_title="Wikipedia QA Chat",
        page_icon="📚"
    )

    st.title("_Wikipedia 기반 :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 위키피디아 기반 QA 챗봇입니다. 궁금한 주제를 입력하고 질문해보세요!"}]

    # ----- 사이드바 -----
    with st.sidebar:
        keyword = st.text_input("📚 Wikipedia 키워드 입력", key="wiki_keyword")
        openai_api_key = st.text_input("🔑 OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("🔄 검색 및 임베딩")

    # ----- Wikipedia 처리 -----
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        if not keyword:
            st.info("Wikipedia 키워드를 입력하세요.")
            st.stop()

        wiki_text = get_text_from_wikipedia(keyword)
        if not wiki_text:
            st.warning("Wikipedia에서 문서를 찾을 수 없습니다.")
            st.stop()

        documents = [Document(page_content=wiki_text, metadata={"source": f"https://ko.wikipedia.org/wiki/{keyword}"})]
        text_chunks = get_text_chunks(documents)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    # ----- 이전 대화 출력 -----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # ----- 유저 질문 처리 -----
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']

                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("📄 참고한 위키 문서"):
                    for doc in source_documents:
                        st.markdown(f"[{doc.metadata['source']}]({doc.metadata['source']})")

        st.session_state.messages.append({"role": "assistant", "content": response})


# ----- 함수: Wikipedia에서 문서 가져오기 -----
def get_text_from_wikipedia(keyword, lang='ko'):
    wikipedia.set_lang(lang)
    try:
        summary = wikipedia.page(keyword).content
        return summary
    except Exception as e:
        logger.error(f"Wikipedia 검색 실패: {e}")
        return None


# ----- 함수: 텍스트를 토큰 수 기준으로 자르기 -----
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(text_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text_docs)
    return chunks


# ----- 함수: 텍스트를 벡터스토어로 변환 -----
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


# ----- 함수: Conversational RAG 체인 만들기 -----
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain


# ----- 실행 -----
if __name__ == '__main__':
    main()

