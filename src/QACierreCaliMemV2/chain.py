from langchain_core.prompts import  ChatPromptTemplate
from langchain_groq import ChatGroq

from sqlalchemy import create_engine
from sqlalchemy import text


from langchain_community.chat_message_histories import ChatMessageHistory


from dotenv import load_dotenv

import sys

load_dotenv()

sys.path.append(".")

# Configuración del modelo
llm = ChatGroq(model="mixtral-8x7b-32768")
#configurar base de datos
engine = create_engine("sqlite:///data/sqlite/cierre.db")
query= """select FECHA_CIERRE, TIME(SUM( DURACION), 'unixepoch') as DURACION_TOTAL_CIERRE,  TIME(SUM(IIF(CODIGO_TAREA=='PAUSA',0,DURACION)),'unixepoch') AS DURACION_CIERRE_SIN_PAUSAS,
                        DATETIME( min(INICIO)) as INICIO_CIERRE, DATETIME( max(FIN)) as FIN_CIERRE, DATETIME(MAX(IIF(DESCRIPCION_TAREA='Habilita accesos al menu',FIN,0))) AS FECHA_HABILITAR_MENU
                    from Cierre c 
                    WHERE FECHA_CIERRE IS NOT NULL
                    group by FECHA_CIERRE 
                  """



context= ""
with engine.connect() as connection:
    result = connection.execute(text(query))
    for row in result:
        context += f"(FECHA_CIERRE={row[0]}, DURACION_TOTAL={row[1]}, DURACION_SIN_PAUSAS={row[2]}, INICIO_CIERRE={row[3]}, FIN_CIERRE={row[4]}, HORA_HABILITAR_MENU={row[5]}); "



#Clase para gestionar memoria

history = ChatMessageHistory()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente muy útil. Por favor entraga solamente la respuesta a la pregunta de manera concreta.
               La respuesta es para altos ejecutivos que no concen el modelo de datos, 
               Aquí tienes la información sobre los procesos de cierre con los siguientes datos 
               FECHA_CIERRE: fecha del cierre
               DURACION_TOTAL: duración de todo el proceso de cierre de cada fecha
               DURACION_SIN_PAUSAS: duración de las tareas de cierre sin contar las pausas
               INICIO_CIERRE: Fecha y hora de inicio del cierre
               FIN_CIERRE: Fecha y hora de fin de todo del cierre
               HORA_HABILITAR_MENU: Fecha y hora en que finalizó la tarea de habilitar menú lo que permite abrir oficinas
            
            """+ context + " no utilices los nombres de los campos en la respuesta, utiliza un lenguaje para ejecutivo",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

from langchain_core.runnables.history import RunnableWithMessageHistory


chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def get_history() -> ChatMessageHistory:
    return history
    

def get_response(user_input):

    
    response =chain_with_message_history.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "unused"}}
    )
    print(response)
    return response.content


def main():
    user_input = ""
    while True:
        user_input=input("Ingrese la pregunta (/q para finalizar)")
        if user_input.startswith("/q"):
            break
        response = get_response(user_input)
        print(response)


if __name__ == "__main__":
    main()