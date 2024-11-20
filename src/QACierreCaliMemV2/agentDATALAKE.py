from datetime import date
import pandas as pd
from langchain_groq  import ChatGroq
from langchain_openai   import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from io import StringIO
import logging
import requests
import os
from dotenv import load_dotenv

# Configure logging level from environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    logging.basicConfig(level=getattr(logging, log_level),format='%(asctime)s %(levelname)s %(message)s')
   
except ValueError:
    logging.basicConfig(level=logging.INFO)
    logging.warning(f"Invalid LOG_LEVEL '{os.environ.get('LOG_LEVEL', 'INFO')}', using INFO instead.")


logging.info(f'cargando modulo - log_level{log_level}')

load_dotenv()



def obtener_datos_cierre_normativo() ->str:
    """Retorna los datos del cierre normativo para todas las fechas disponibles específica en formato json
        con los siguientes campos:
               FECHA_CIERRE: fecha del cierre
               FIN_BANDEJA4: fecha y hora en la que está disponible la información para enviar a la Super Intendencia Financiera
               FIN_BANDEJA8: fecha y hora en la finaliza todo el cierre normativo

        Args:
            fecha_cierre: fecha de cierre, Opcional. 
    """
    try:
        myurl = os.environ["RESUMEN_CIERE_NORMATIVO_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]

        response = requests.get(myurl)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        context = response.text
        return context
    except KeyError as e:
        logging.error(f"Error with env variables RESUMEN_CIERE_NORMATIVO_URL or KEY")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from RESUMEN_CIERE_NORMATIVO_URL: {e}")
        return None


def df_cierre_comercial():
    #Leer archivo
    try:
        myurl = os.environ["RESUMEN_CIERE_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]
   
        response = requests.get(myurl)
        response.raise_for_status()
        context = response.text
        #crear dataframe
        df:pd.DataFrame=pd.read_json(StringIO(context))
        #Formatear fecha y duraciones 
        df["FECHA_CIERRE"]=df.FECHA_CIERRE.str.slice(0,10)
        df["Duración Total"]= df.DURACION_TOTAL_CIERRE_SEGUNDOS/60/60
        df["Sin Pausas"]= df.DURACION_SIN_PAUSAS_SEGUNDOS/60/60
        df["Total Pausas"]=  df["Duración Total"]-df["Sin Pausas"]
        return df
    except (requests.exceptions.RequestException, pd.errors.EmptyDataError, KeyError) as e:
        logging.error(f"Error processing cierre comercial data: {e}")
        return None


def obtener_datos_cierre_comercial() ->str:
    """Retorna los datos del proceso de cierre comercial para todas las fechas disponibles específica en formato json
        con los siguientes campos:
               FECHA_CIERRE: fecha del cierre
               DURACION_TOTAL: duración de todo el proceso de cierre de cada fecha
               DURACION_SIN_PAUSAS: duración de las tareas de cierre sin contar las pausas
               INICIO_CIERRE: Fecha y hora de inicio del cierre
               FIN_CIERRE: Fecha y hora de fin de todo del cierre
               HORA_HABILITAR_MENU: Fecha y hora en que finalizó la tarea de habilitar menú lo que permite abrir oficinas. Cuando de este dato puntualice si se logró antes de las 8:00 am o no
        Args:
            fecha_cierre: fecha de cierre, Opcional. 
    """
    myurl = os.environ["RESUMEN_CIERRE_URL"]+"?"+os.environ["AZURE_DATALAKE_GENAI_TOKEN"]
    try:
        response = requests.get(myurl)
        response.raise_for_status()
        context = response.text
        return context
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {myurl}: {e}")
        return None


def obtener_datos_tareas_mayor_duracion_por_fecha(fecha_cierre:date) ->str:
    """Retorna los detalles las tareas con mayor duración no tienen información del cierre completo 
    solamente de las 10 tareas de mayor duracion del cierre comercial
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
    try:
        df=pd.read_parquet(myurl)
        context= df[df.FECHA_CIERRE == fecha_cierre].sort_values(by='DURACION_SEGUNDOS' ,ascending=False).head(10).to_json(orient='records')
        return context
    except (pd.errors.EmptyDataError, KeyError, FileNotFoundError) as e:
        logging.error(f"Error fetching or processing task data for {fecha_cierre}: {e}")
        return None



#### Configuración del modelo y herramientas
llm= AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    ,azure_deployment= os.environ["AZURE_OPENAI_DEPLOYMENT"]
    ,temperature=0
    ,verbose=False)

tools=[obtener_datos_cierre_comercial, obtener_datos_tareas_mayor_duracion_por_fecha, obtener_datos_cierre_normativo]

fecha_hoy=f" La fecha actual es {date.today().strftime('%Y-%m-%d')}"

## Configuración del agente
system_message = """"Eres un asistente muy útil y tienes acceso a la información de proceso de cierre comercial y cierre normativo.
              cuando se pregunta especificamente por cierre comercial o normativo, asume que es refieren al cierre comercial.
              por favor utiliza respuestas gerenciales y concisas. 
              No respondas preguntas sobre temas diferentes al cierre comercial y el normativo.
            """ + fecha_hoy
system_message_1shot = """"Eres un asistente muy útil y tienes acceso a la información de proceso de cierre comercial y cierre normativo.
              cuando se pregunta especificamente por cierre comercial o normativo, asume que es refieren al cierre comercial.
              por favor utiliza respuestas gerenciales y concisas. Por ejemplo:
              Ejemplo 1:"El cierre comercial del FECHA_CIERRE inició a las (fecha y hora) y finalizó a las (fecha y hora). La duración total fueron (DURACION) y 
              la duración sin pausas fue (DURACION_SIN_PAUSAS). La tarea de habilitar menú finalizó a las (HORA_HABILITAR_MENU), dado que finalizó antes de las 8:00 am
              no impacto la apertura de oficinas"
              Ejemplo 2:"El cierre normativo finalizó completamente a las (FIN_BANDEJA8). La información para enviar a las Super Intendencia estuvo disponible a las (FIN_BANDEJA4)"              
              Responde siempre en español
              No respondas preguntas sobre temas diferentes al cierre comercial y el normativo
            """ + fecha_hoy

memory = MemorySaver()

agent_executor = create_react_agent(
    llm, tools=tools, state_modifier=system_message_1shot,debug=False, checkpointer=memory
)

#Función para obtener última respuesta del agente
def get_response(user_input,thread_id):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [("user", user_input)]}
        response = agent_executor.invoke(inputs, config=config)
        for m in response["messages"]:
            logging.debug(f"{m}")
        message = response["messages"][-1]
        if isinstance(message,tuple):
            return(message[1])
        else:
            return message.content
    except Exception as e:
        logging.exception(f"An error occurred during agent execution: {e}")
        return "An unexpected error occurred."


def main():
    config = {"configurable": {"thread_id": "1"}}
    user_input = ""
    while True:
        user_input=input("Ingrese la pregunta (/q para finalizar): \n")
        if user_input.startswith("/q"):
            break
        try:
            inputs = {"messages": [("user", user_input)]}
            response = agent_executor.stream(inputs,config=config)
            for s in response:
                for key in s.keys():
                    message=s[key]['messages'][-1]
                    if isinstance(message,tuple):
                        print(message)
                    else:
                        message.pretty_print()
        except Exception as e:
            logging.exception(f"An error occurred during streaming response: {e}")


if __name__ == "__main__":
    main()
