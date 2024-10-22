import streamlit as st

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

import sys

load_dotenv()

sys.path.append(".")
from src.QACierreCaliMemV2.agentDATALAKE import get_response

# Inicializar la memoria de chat
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMessageHistory()

# Configuración de la aplicación
st.title("Consulta Cierre 10 últimos días")

# Mostrar el historial de la conversación
def display_chat(history: ChatMessageHistory):
    for message in history.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        else:
            with st.chat_message("user",):
                st.markdown(message.content)
            


# Área de entrada de texto

display_chat(st.session_state.chat_memory)

if prompt := st.chat_input("Pregunta"):
    if prompt:
        # Obtener respuesta del modelo
        st.session_state.chat_memory.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status("Consultando información ..." , expanded=True) as status:
                
            response = get_response(prompt)
            st.session_state.chat_memory.add_ai_message(response)
        # Mostrar la respuesta
            with st.chat_message("assistant"):
                st.markdown(response) 
            status.update(label="",state="complete")      

        
            # Limpiar el campo de entrada


