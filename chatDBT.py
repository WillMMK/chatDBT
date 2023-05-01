from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import tiktoken
import os 
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent, Tool


# Check if the embedded files directory exists
embeddings = OpenAIEmbeddings()
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not os.path.exists('embedded_files/dbt/'):
    # load the document loader and the embeddings
    loader = DirectoryLoader('docs.getdbt.com/docs', glob="*.html")
    pages = loader.load()
    # we need to calculate the average number of tokens in the pages
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def calculate_average_num_tokens(pages):
        num_tokens_list = []
        for page in pages:
            num_tokens = num_tokens_from_string(page.page_content, "cl100k_base")
            num_tokens_list.append(num_tokens)
        return int(sum(num_tokens_list) / len(num_tokens_list))
    
    average = calculate_average_num_tokens(pages)
    
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Chroma, 
        embedding=OpenAIEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=average, chunk_overlap=0)
    )

    docsearch = Chroma.from_documents(pages, embeddings, collection_name="dbt",persist_directory="embedded_files/dbt/")
    docsearch.persist()
else:
    # Now we can load the persisted database from disk, and use it as normal. 
    docsearch = Chroma(persist_directory='embedded_files/dbt/', embedding_function=embeddings,collection_name="dbt")

retriever = docsearch.as_retriever()

# Initialize the chatbot
def initialize_dbt_chatbot():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key,
        max_tokens=200,
    )

    dbt_template = """You are a useful and friendly chatbot provide accurate and helpful information and example about dbt.Do not make thing up.
    If you do no know, say I do not know.
    {context}

    Question: {question}"""
    dbt_PROMPT = PromptTemplate(
        template=dbt_template, input_variables=["context", "question"]
    )
    dbt_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": dbt_PROMPT}
    )

    tools = [
        Tool(
            name = "dbt_chatbot",
            func=dbt_qa.run,
            description="""useful for when a user is interested in dbt information.
                        Input should be a fully formed question."""
        )]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, max_iterations=5)
    return agent 
agent = initialize_dbt_chatbot()

while True:
    question = input("Hi, I am chatDBT! Ask me any question about dbt or type 'quit' to exit: ")
    
    if question.lower() == 'quit':
        print("Goodbye! If you have more questions in the future, feel free to ask.")
        break
    
    agent.run(question)