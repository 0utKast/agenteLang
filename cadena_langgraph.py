import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- Importaciones de LangChain para construir los nodos ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PASO 1: Cargar la clave de API desde .env ---
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY no encontrada. Asegúrate de crear el archivo .env")
    exit()

# --- PASO 2: Definir el "Estado" del Grafo ---
# El 'Estado' es un objeto (aquí, un diccionario) que se pasa
# entre todos los nodos. Es la "memoria" de nuestro agente.
# TypedDict nos ayuda a definir su estructura.
class AgentState(TypedDict):
    question: str
    answer: str
    attempts: int

# --- PASO 3: Definir los "Nodos" (Funciones) ---

# Nodo 1: Generar una respuesta
def generate_answer(state: AgentState):
    """
    Toma la pregunta del estado actual, llama al LLM para generar
    una respuesta y actualiza el estado con esa respuesta.
    """
    print(f"--- Agente: Generando respuesta... (Intento {state['attempts'] + 1}) ---")
    
    # 1. Obtenemos la pregunta del estado
    question = state["question"]
    
    # 2. Configuramos el LLM (Gemini) con el nombre completo del modelo
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")
    
    # 3. Creamos un prompt para guiar al LLM
    prompt = ChatPromptTemplate.from_template(
        "Responde a la siguiente pregunta: {question}. "
        "Por favor, sé educado y asegúrate de incluir la palabra 'gracias' en tu respuesta."
    )
    
    # 4. Creamos una mini-cadena (LCEL) solo para este nodo
    chain = prompt | llm | StrOutputParser()
    
    # 5. Invocamos la cadena
    generated_answer = chain.invoke({"question": question})
    
    # 6. Devolvemos un diccionario con los campos del estado que queremos actualizar
    return {
        "answer": generated_answer,
        "attempts": state["attempts"] + 1
    }

# Nodo 2: Función de decisión (será una 'Arista Condicional')
def decide_next_step(state: AgentState):
    """
    Revisa la respuesta en el estado y decide a qué nodo
    ir a continuación. Esto crea los bucles y la lógica.
    """
    answer = state["answer"]
    print(f"--- Agente: Revisando respuesta: '{answer}' ---")

    # Condición de ejemplo: la respuesta DEBE contener "unicornio"
    if "unicornio" in answer.lower(): # <--- LÍNEA MODIFICADA
        print("--- Agente: Respuesta validada. Contiene 'unicornio'. ---")
        # Si se aprueba, devolvemos la cadena 'approved', que apunta a END
        return "approved"
    else:
        print("--- Agente: Respuesta RECHAZADA. No contiene 'unicornio'. ---") # <--- MODIFICADO PARA CLARIDAD
        
        # Lógica de reintento: si hemos intentado demasiado, paramos.
        if state["attempts"] >= 3:
            print("--- Agente: Demasiados intentos. Finalizando. ---")
            return "fallback_end"
        else:
            # Si no, devolvemos 're_attempt', que apunta de nuevo a 'generate'
            print("--- Agente: Reintentando... ---")
            return "re_attempt"


# --- PASO 4: Construir el Grafo ---

# 4.1 Inicializar el constructor del grafo
workflow = StateGraph(AgentState)

# 4.2 Añadir los nodos al grafo
# El primer argumento es un nombre único, el segundo es la función
workflow.add_node("generate", generate_answer)
# Nota: 'decide_next_step' no es un nodo, es la lógica de una arista

# 4.3 Definir el punto de entrada
# Le decimos al grafo por dónde debe empezar
workflow.set_entry_point("generate")

# 4.4 Añadir las aristas (las conexiones)
# Aquí está la magia de LangGraph: Aristas Condicionales
workflow.add_conditional_edges(
    "generate",          # Nodo de origen: después de generar...
    decide_next_step,    # Función de decisión: ...ejecuta esta función...
    {
        # ...y según lo que devuelva la función, ve a uno de estos nodos:
        "approved": END,          # Si devuelve "approved", termina el grafo
        "re_attempt": "generate", # Si devuelve "re_attempt", vuelve al nodo 'generate' (¡el bucle!)
        "fallback_end": END       # Si devuelve "fallback_end", termina
    }
)

# 4.5 Compilar el grafo
# Se "congela" la estructura del grafo en un objeto ejecutable (Runnable)
app = workflow.compile()

# --- PASO 5: Ejecutar el Grafo ---
print("\n--- LangGraph con Gemini: Agente de Auto-corrección ---")
# El estado inicial que le pasamos al grafo
inputs = {"question": "¿Cómo está el clima hoy?", "attempts": 0, "answer": ""}

# Usamos 'stream' para ver cada paso (cada nodo) a medida que se ejecuta
for s in app.stream(inputs, {"recursion_limit": 5}): # Límite de recursión por seguridad
    print(s)
    print("--------------------")