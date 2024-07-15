import streamlit as st
from zhipuai_llm import ZhipuAILLM
import os
import re
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file

zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']

def generate_response(input_text, zhipuai_api_key):
    llm = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipuai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output

def get_split_docs():
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
    loader = PyMuPDFLoader("electricity_data.pdf")
    # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
    pdf_pages = loader.load()

    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    i = 0
    for pdf_page in pdf_pages:
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
        pdf_pages[i] = pdf_page
        i += 1
    # print(pdf_page.page_content)

    # 知识库中单段文本长度
    CHUNK_SIZE = 500
    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=OVERLAP_SIZE)
    for pdf_page in pdf_pages:
        text_splitter.split_text(pdf_page.page_content)
        # print(pdf_page)
    split_docs = text_splitter.split_documents(pdf_pages)
    return split_docs


def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    split_docs = get_split_docs()
    # 加载数据库
    vectordb = FAISS.from_documents(
        split_docs,
        embedding
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,zhipuai_api_key:str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipuai_api_key)              
    # 历史消息
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True  # 将以消息列表的形式返回聊天记录，
    )
    retriever=vectordb.as_retriever()
    # 生成回答
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,zhipuai_api_key:str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = zhipuai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 应用程序界面
def main():
    st.title('动手学大模型应用开发')
    zhipuai_api_key = st.sidebar.text_input('输入你的ZHIPUAI_API_KEY', type='password')

    # 添加一个选择按钮来选择不同的模型
    selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio("你想选择哪种模式进行对话？",["None", "qa_chain", "chat_qa_chain"],
                      captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 初始化会话状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})
        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, zhipuai_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,zhipuai_api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,zhipuai_api_key)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

if __name__ == "__main__":
    main()
