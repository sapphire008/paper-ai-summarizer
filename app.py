import streamlit as st
from dynaconf import Dynaconf
from chat import initialize_conversation

def load_settings(settings_file="settings.yaml"):
    settings = Dynaconf(
        settings_file=settings_file,
        environments=True,
    )
    return settings

def main(settings={}):
    # Set title
    st.title(settings.get("APP_TITLE", "Research Assistant"))

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Side bar to handle document uploads
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )
        st.write(f"Uploaded {len(pdf_docs)} file(s)")

        if st.button("Process"):
            # Start the conversation only when documents are uploaded
            with st.spinner("Processing"):
                st.session_state.conversation = initialize_conversation(pdf_docs, settings)

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get the response from our conversation chain
        response = st.session_state.conversation({"question": prompt})
        response = response["chat_history"][-1].content
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    settings = load_settings()
    main(settings)