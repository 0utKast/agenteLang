import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PASO 1: Cargar la clave de API desde .env ---
# Carga las variables del archivo .env en el entorno del sistema.
load_dotenv()

# Comprobación de seguridad para asegurar que la clave de API se ha cargado.
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY no encontrada. Asegúrate de crear el archivo .env")
    exit()

# --- PASO 2: Configurar el LLM (el 'cerebro') ---
# Se instancia el modelo de chat de Google.
# 'model' especifica qué versión de Gemini usar.
# 'temperature=0' asegura respuestas deterministas (no creativas).
llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0)


# --- PASO 3: Crear la Plantilla de Prompt ---
# Se define la plantilla de instrucciones que recibirá el LLM.
# "system" define el rol o la instrucción base del asistente.
# "user" define el formato de la entrada del usuario, con variables.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil. Responde a la pregunta basada únicamente en el contexto proporcionado:"),
    ("user", "{context}\n\nPregunta: {input}")
])

# --- PASO 4: Definir el Analizador de Salida ---
# Se instancia un analizador simple que convierte la salida del LLM
# (que es un objeto complejo) en una simple cadena de texto (string).
output_parser = StrOutputParser()

# --- PASO 5: Ensamblar y Ejecutar la Cadena ---

# 5.1 Ensamblado de la Cadena (LCEL)
# Se utiliza el LangChain Expression Language (LCEL) con el operador '|' (pipe).
# Esto crea una secuencia (cadena) donde la salida de un componente
# es la entrada del siguiente.
# El flujo es: Diccionario de entrada -> prompt_template -> llm -> output_parser
chain = prompt_template | llm | output_parser

# 5.2 Ejecución de la Cadena
# Se definen las variables de entrada que llenarán el prompt.
contexto_ejemplo = "El equipo de fútbol ganó el campeonato en 2023. Su delantero estrella, Marco, marcó el gol decisivo en la final."
pregunta_ejemplo = "¿Quién marcó el gol decisivo?"

print("--- LangChain Básico con Gemini ---")
# 'invoke' es el método para ejecutar la cadena.
# Le pasamos un diccionario que coincide con las variables del prompt.
respuesta = chain.invoke({"context": contexto_ejemplo, "input": pregunta_ejemplo})

# Se imprime el resultado final (ya procesado por el StrOutputParser).
print(f"Pregunta: {pregunta_ejemplo}")
print(f"Respuesta: {respuesta}")