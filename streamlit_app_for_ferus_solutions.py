import os
import time
from langchain import ConversationChain, LLMChain
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
try:
    import environment_variables
except ImportError:
    pass


def sanitize_answer(raw_answer):
    answer = ''
    chunks = raw_answer.split('\n')
    sub_string_list = []
    sub_string_list.append('- SOURCE:')
    sub_string_list.append('- Source:')
    sub_string_list.append('- source:')
    sub_string_list.append('- SOURCES:')
    sub_string_list.append('- Sources:')
    sub_string_list.append('- sources:')
    sub_string_list.append('(SOURCE:')
    sub_string_list.append('(Source:')
    sub_string_list.append('(source:')
    sub_string_list.append('(SOURCES:')
    sub_string_list.append('(Sources:')
    sub_string_list.append('(sources:')
    sub_string_list.append('SOURCE:')
    sub_string_list.append('Source:')
    sub_string_list.append('source:')
    sub_string_list.append('SOURCES:')
    sub_string_list.append('Sources:')
    sub_string_list.append('sources:')

    try:
        for chunk in chunks:
            temp_string = str(chunk)
            temp_string = temp_string.strip()
            temp_string_lowercase = temp_string.lower()
            answer_text = ''

            if temp_string_lowercase.find('source') != -1:
                for sub_string in sub_string_list:
                    if temp_string.find(sub_string) != -1:
                        # print(f'{sub_string} - {temp_string}')
                        temp_string = temp_string[:temp_string.index(
                            sub_string)]

            # Append answer text
            answer_text = temp_string.strip()
            if len(answer_text) > 0:
                # print(answer_text)
                answer = answer + '\n\n' + answer_text

        answer = answer.strip()
    except Exception as e:
        error_message = ''
        # st.text('Hello World')
        st.error('An error has occurred. Please try again.', icon="🚨")
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            error_message = e.message
        else:
            error_message = e
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # st.error('ERROR MESSAGE: {}'.format(error_message))

    return answer


try:
    # Setting page title and header
    st.set_page_config(page_title="AI ChatBot", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>AI ChatBot 😬</h1>",
                unsafe_allow_html=True)

    # Get environment variables
    # openai.organization = os.environ['openai_organization']
    # =======================================================
    OPENAI_API_KEY = os.environ['openai_api_key']
    pinecone_api_key = os.environ['pinecone_api_key']
    pinecone_environment = os.environ['pinecone_environment']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    index_name = os.environ['index_name']
    # ==================================================== #

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-4"

    # Initialize the large language model
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name=model,
    )

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['model_name'] = []
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

    def get_prompt():
        # Define the system message template
        system_template = """ 
        Answer the question into bullet point list.
        """

        # user_template = "Question:```{question}```"

        # Create the chat prompt templates
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            # HumanMessagePromptTemplate.from_template(user_template),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        return prompt

    # generate a response
    def generate_response(prompt):
        query = prompt
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        # PROMPT = get_prompt()

        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=16384,
            input_key='question',
            output_key='answer',
        )

        # conversation = ConversationChain(
        #     prompt=PROMPT,
        #     llm=llm,
        #     verbose=True,
        #     memory=memory,
        # )

        # response = conversation.run(prompt)

        # chain = ConversationalRetrievalChain.from_llm(
        #     llm,
        #     combine_docs_chain_kwargs={"prompt": PROMPT},
        #     memory=memory,
        #     return_source_documents=True,
        #     verbose=False,
        # )
        # response = chain.run({'question': query})
        # child_response = response.strip()

        # child_query = f'{child_query}. Create a bulleted list.'
        # st.sidebar.text(child_query)

        # result = chain({'question': query})
        # raw_answer = result['answer']
        # response = sanitize_answer(raw_answer)

        # conversation = ConversationChain(
        #     llm=llm,
        #     # memory=memory,
        #     prompt=PROMPT,
        #     verbose=True
        # )
        # response = conversation.run({'input': query})

        template = """

        The following is a friendly conversation between a human and an AI. The topic is breast cance.
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does
        not know.

        Current conversation:
        Human: Give me detailed information about breast cancer {input}
        AI Assistant:"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template)

        example_human_history = HumanMessagePromptTemplate.from_template("Hi")
        example_ai_history = AIMessagePromptTemplate.from_template(
            "hello, how are you today?")

        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, example_human_history, example_ai_history, human_message_prompt])

        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        chain = LLMChain(llm=chat, prompt=chat_prompt)

        response = chain.run(query)

        ######################################################
        st.session_state['messages'].append(
            {"role": "assistant", "content": response})

        return response

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            # user_input = st.text_area("You:", key='input', height=100)
            user_input = st.selectbox(
                "What do you want to know about breast cancer?",
                ("Treatment", "Symptoms", "Causes", "Types",
                 "Risk factors", "Prevention", "Diagnosis",),
                key='input',
            )
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = generate_response(
                user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(f'Give me detailed information about breast cancer {st.session_state["past"][i]}',
                        is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
    # Just print(e) is cleaner and more likely what you want,
    # but if you insist on printing message specifically whenever possible...
    if hasattr(e, 'message'):
        error_message = e.message
    else:
        error_message = e
    st.error('ERROR MESSAGE: {}'.format(error_message))
