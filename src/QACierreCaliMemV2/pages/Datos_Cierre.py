import streamlit as st
import uuid
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

import sys

load_dotenv()

sys.path.append(".")
from src.QACierreCaliMemV2.agentDATALAKE import get_response, df_cierre_comercial

st.set_page_config(layout="wide")
# Inicializar la memoria de chat
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMessageHistory()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# Configuración de la aplicación
st.title("Consulta Cierre 30 últimos días")
st.divider()
col1, col2 = st.columns([0.5,0.5],gap='medium')
# Mostrar el historial de la conversación
def display_chat(history: ChatMessageHistory):
    for message in history.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        else:
            with st.chat_message("user",):
                st.markdown(message.content)
            

with col1:
    df=df_cierre_comercial()
    st.bar_chart(df,x="FECHA_CIERRE",y=["Duración Total","Sin Pausas"],stack=False, y_label="Duración (hrs)")
    #st.bar_chart(df,x="FECHA_CIERRE",y=["Sin Pausas","Total Pausas"],stack=True, y_label="Duración (hrs)")
with col2:
    chat_container = st.container(height=500)
    with chat_container:
        display_chat(st.session_state.chat_memory)

    if prompt := st.chat_input("Pregunta"):
        if prompt:
            # Obtener respuesta del modelo
            st.session_state.chat_memory.add_user_message(prompt)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.status("Consultando información ..." , expanded=True) as status:
                    
                    response = get_response(prompt,st.session_state["thread_id"])
                    st.session_state.chat_memory.add_ai_message(response)
            # Mostrar la respuesta
                    with st.chat_message("assistant"):
                        st.markdown(response) 
                        status.update(label="",state="complete")      

            
            # Limpiar el campo de entrada
