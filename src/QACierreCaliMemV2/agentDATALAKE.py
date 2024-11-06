from datetime import date
import pandas as pd
from langchain_groq  import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import logging
#from langchain_community.utilities.sql_database import SQLDatabase
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

import requests
import os

from dotenv import load_dotenv


logging.basicConfig(filename='logs/app.log', level=logging.INFO)

logging.info('cargando modulo')

load_dotenv()

#Conexión de base de datos
#engine = create_engine("sqlite:///data/sqlite/cierre.db")



def obtener_datos_cierre_normativo() ->str:
    """Retorna los datos del cierre normativo para todas las fechas disponibles específica en formato json
        con los siguientes campos:
               FECHA_CIERRE: fecha del cierre
               FIN_BANDEJA4: fecha y hora de finalización del proceso bandeja 4
               FIN_BANDEJA8: fecha y hora de finalización del proceso bandeja 8 que es igual a la finalización del cierre normativo
        Args:
            fecha_cierre: fecha de cierre, Opcional. 
    """
  
    myurl = os.environ["RESUMEN_CIERE_NORMATIVO_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]
    #myfile = urlopen(myurl)
    context = requests.get(myurl).text
    
    
    return context

def obtener_datos_por_proceso_de_cierre() ->str:
    """Retorna los datos del proceso de cierre para todas las fechas disponibles específica en formato json
        con los siguientes campos:
               FECHA_CIERRE: fecha del cierre
               DURACION_TOTAL: duración de todo el proceso de cierre de cada fecha
               DURACION_SIN_PAUSAS: duración de las tareas de cierre sin contar las pausas
               INICIO_CIERRE: Fecha y hora de inicio del cierre
               FIN_CIERRE: Fecha y hora de fin de todo del cierre
               HORA_HABILITAR_MENU: Fecha y hora en que finalizó la tarea de habilitar menú lo que permite abrir oficinas
        Args:
            fecha_cierre: fecha de cierre, Opcional. 
    """
  
    myurl = os.environ["RESUMEN_CIERE_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]
    #myfile = urlopen(myurl)
    context = requests.get(myurl).text
    
    
    return context

###########
# funciones para ser invocadas por el modelo
def obtener_datos_tareas_mayor_duracion_por_fecha(fecha_cierre:date) ->str:
    """Retorna los detalles las tareas con mayor duración no tienen información del cierre completo 
    solamente de las 10 tareas de mayor duracion
        los datos disponibles enen formato json son:
               FECHA_CIERRE: fecha del cierre
               DURACION: duración de la tarea en ejecución
               INICIO: Fecha y hora de inicio de la tarea
               FIN: Fecha y hora de fin de la tarea
               CODIGO_TAREA: Identificador de la tarea
               DESCRIPCION_TAREA: Descripción que complementa el código de la tarea
        Args:
            fecha_cierre: fecha de cierre 
    """
    myurl = os.environ["DETALLE_TAREAS_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]
    df=pd.read_parquet(myurl)
    context= df[df.FECHA_CIERRE == fecha_cierre].sort_values(by='DURACION_SEGUNDOS' ,ascending=False).head(10).to_json(orient='records')
    return context



#### Configuración del modelo y herramientas

#grop_model = "gemma-7b-it"  -- NO
#grop_model ="mixtral-8x7b-32768"
#
grop_model ="llama3-groq-70b-8192-tool-use-preview"

llm = ChatGroq(model=grop_model, temperature=0,verbose=True)

tools=[obtener_datos_por_proceso_de_cierre, obtener_datos_tareas_mayor_duracion_por_fecha]






## Configuración del agente
system_message = """"Eres un asistente muy útil. Por favor entraga solamente la respuesta a la pregunta de manera concreta.
              identifica primero si la pregunta es sobre todo el cierre o sobre las tareas de mayor duración para elegir la herramienta (tool) más adecuada
              para responder sobre las tareas de mayor duración de una fecha especifica, primero valide si hay datos para esa fecha de cierre
              Responde siempre en español
            """

memory = MemorySaver()

agent_executor = create_react_agent(
    llm, tools=tools, state_modifier=system_message,debug=False, checkpointer=memory
)


def get_response(user_input,thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", user_input)]}
    response = agent_executor.invoke(inputs, config=config)
    for m in response["messages"]:
            print("print:",m)
            logging.info(f"info: {m}")

    
    message = response["messages"][-1]
    if isinstance(message,tuple):
        return(message[1])
    else:
        return message.content
           

def main():
    user_input = ""
    while True:
        user_input=input("Ingrese la pregunta (/q para finalizar): \n")
        if user_input.startswith("/q"):
            break

        inputs = {"messages": [("user", user_input)]}
        response = agent_executor.stream(inputs,config=config)
     #   print(type(response))
        for s in response:
            for key in s.keys():
                message=s[key]['messages'][-1]
                if isinstance(message,tuple):
                    print(message)
                else:
                    message.pretty_print()
           

if __name__ == "__main__":
    main()