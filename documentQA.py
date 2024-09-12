import os
import streamlit as st
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone

###Load OpenAI
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)

###Load pinecone
PINECONE_KEY=os.environ['PINECONE_KEY']
PINECONE_INDEX=os.environ['PINECONE_INDEX']
pc = Pinecone(PINECONE_KEY)
index=pc.Index(host=PINECONE_INDEX)


###Query Embedding Generation
def query_emebeddings(query):
    embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    query_embedding=embeddings.embed_query(str(query))
    return query_embedding

###Top K(k=5) retrieval with similarity scores
def retrieve_articles(query):
    query_embed = query_emebeddings(query)

    results = index.query(
        namespace="ns1",
        vector=query_embed,
        top_k=5,
        include_values=False,
        include_metadata=True
    )

    # print(results)
    data = []
    for i in results['matches']:
        content = i['metadata']['text']
        score = i['score']
        data.append({"content": content, "score": score})

    return data

###Answer Generation
def answer_generation(query):
    content=retrieve_articles(query)
    ###Dynamic Input Prompt
    def summarization_prompt(article_content, query):
        prompt = f"""

        Matching Article content along with similarity scores:
        {article_content}

        query:{query}

        Generate a comprehensive answer for the above given query based on the articles given

        """
        return prompt
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": summarization_prompt(content,query),
            }
        ],
        model="gpt-3.5-turbo",
        temperature=0.0
    )

    return response.choices[0].message.content


###Initialize streamlit app

st.set_page_config(page_title='RAG Application')
st.header('Document Q&A')

input=st.text_input("Question:",key='input')
##response
output=answer_generation(input)
submit=st.button("Submit")

if submit:
    st.subheader('The response is')
    st.write(output)




