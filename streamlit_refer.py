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
        page_title="질문 기반 Wikipedia QA Chat",
        page_icon="📚"
    )

    st.title("_Wikipedia 기반 :red[질문형 QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 궁금한 내용을 질문해주세요. 관련 Wikipedia 문서를 검색해서 답변해드릴게요."}]

    with st.sidebar:
        openai_api_key = st.text_input("🔑 OpenAI API Key", key="chatbot_api_key", type="password")

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 질문 입력
    if query := st.chat_input("궁금한 점을 입력하세요. 예: 대한민국의 수도는 어디인가요?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not openai_api_key:
                st.warning("OpenAI API 키를 입력해주세요.")
                st.stop()

            # 🔍 질문 → 위키 문서 검색
            title, wiki_text = search_wikipedia_from_question(query)
            if not wiki_text:
                st.warning("Wikipedia에서 관련 문서를 찾을 수 없습니다.")
                st.stop()

            documents = [Document(page_content=wiki_text, metadata={"source": f"https://ko.wikipedia.org/wiki/{title}"})]
            text_chunks = get_text_chunks(documents)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": query})
                response = result['answer']
                source_documents = result['source_documents']
                st.markdown(response)

                with st.expander("📄 참고한 위키 문서"):
                    for doc in source_documents:
                        st.markdown(f"[{doc.metadata['source']}]({doc.metadata['source']})")

            st.session_state.messages.append({"role": "assistant", "content": response})


# ----- 함수: 질문 → 위키 검색 및 문서 가져오기 -----
def search_wikipedia_from_question(query, lang='ko'):
    wikipedia.set_lang(lang)
    try:
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return None, None
        best_title = search_results[0]
        page = wikipedia.page(best_title)
        return best_title, page.content
    except Exception as e:
        logger.error(f"Wikipedia 검색 실패: {e}")
        return None, None


# ----- 함수: 텍스트 길이 계산 (tiktoken 기반) -----
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# ----- 함수: 문서 chunking -----
def get_text_chunks(text_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text_docs)
    return chunks


# ----- 함수: 벡터스토어 생성 -----
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


# ----- 함수: RAG QA 체인 생성 -----
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


