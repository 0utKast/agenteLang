import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator  # Lo usaremos para acumular mensajes en el estado

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END  # Componentes clave de LangGraph

# --- PASO 1: Configuración Inicial (LLM y API Key) ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

# Instanciamos el LLM que usaremos tanto para el razonamiento como para la herramienta
llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

# --- PASO 2: Definición del Estado del Agente ---
# El Estado es la "memoria" del agente, que persiste y se modifica
# a medida que fluye a través de los nodos del grafo.
class AgentState(TypedDict):
    
    # 'messages': Es el historial de la conversación, el "bloc de notas" (scratchpad).
    # 'Annotated[List, operator.add]' es una característica avanzada de LangGraph.
    # Significa que 'messages' es una Lista, y cada vez que un nodo
    # devuelva un valor para 'messages', se AÑADIRÁ (operator.add) a la lista
    # existente, en lugar de reemplazarla. Así construimos el historial P-A-O.
    messages: Annotated[List, operator.add]
    
    # 'next_action': Un campo simple para que el nodo de razonamiento
    # le diga al "router" (la arista condicional) qué nodo ejecutar a continuación.
    next_action: str
    
    # 'iterations': Un contador simple para evitar bucles infinitos.
    iterations: int

# --- PASO 3: Definición de Herramientas ---
# Para este ejemplo, nuestra "herramienta" es el propio LLM,
# pero llamado de forma específica para responder una sub-pregunta concisa.
def llm_tool(query: str) -> str:
    """Usa el LLM principal para responder una consulta específica."""
    print(f"--- Usando herramienta LLM para la consulta: {query} ---")
    response = llm.invoke(f"Responde esta pregunta de forma concisa: {query}")
    return response.content.strip()

# --- PASO 4: Definición de los Nodos del Grafo ---
# Cada nodo es una función que recibe el estado
# y devuelve un diccionario para actualizar ese estado.

# Nodo 1: El "Cerebro" (Reasoning)
def reasoning_node(state: AgentState):
    """
    El "cerebro" del agente. Lee el historial y decide qué hacer.
    Utiliza un prompt especial para forzar al LLM a seguir el formato ReAct.
    """
    print("\n=== Nodo de Razonamiento ===")
    
    # Control de seguridad para evitar bucles infinitos
    if state["iterations"] >= 5:
        print("--- Límite de iteraciones alcanzado. Forzando finalización. ---")
        return {"next_action": "end"}

    # Obtenemos el historial completo de mensajes
    history = "\n".join(state["messages"])
    
    # Este prompt es CLAVE para el patrón ReAct.
    # Fuerza al LLM a "Pensar" (PENSAMIENTO) y luego a decidir
    # una "Acción" (ACCION: [herramienta] o ACCION: end).
    prompt = f"""
Eres un agente de IA que responde a la pregunta: "{state['messages'][0]}"
La conversación hasta ahora es:
{history}

¿Necesitas usar una herramienta para obtener más información o ya tienes suficiente para responder?
Si necesitas más información, responde SOLAMENTE con:
PENSAMIENTO: [Tu razonamiento sobre qué información necesitas]. ACCION: [La pregunta específica para la herramienta].

Si crees que ya tienes suficiente información para dar una respuesta final, responde SOLAMENTE con:
PENSAMIENTO: [Tu razonamiento de por qué tienes suficiente información]. ACCION: end
"""
    
    # 1. RAZONAR
    response = llm.invoke(prompt)
    decision = response.content.strip()
    print(f"Decisión del LLM: {decision}")

    # 2. DECIDIR
    # Actualizamos el estado con el PENSAMIENTO/ACCION del LLM
    # y (muy importante) fijamos 'next_action' para el router condicional.
    if "ACCION: end" in decision:
        return {"messages": [decision], "next_action": "end"}
    else:
        return {"messages": [decision], "next_action": "action"}

# Nodo 2: Las "Manos" (Action)
def action_node(state: AgentState):
    """
    El nodo de "acción". Se ejecuta si el cerebro decide usar una herramienta.
    Parsea la acción del último mensaje, llama a la herramienta
    y devuelve el resultado como una "OBSERVACION".
    """
    print("\n=== Nodo de Acción ===")
    
    # 1. PARSEAR
    # Obtenemos el último mensaje (que es el PENSAMIENTO/ACCION)
    last_thought = state["messages"][-1]
    
    try:
        # Extraemos la consulta para la herramienta (lo que va después de "ACCION:")
        query = last_thought.split("ACCION:")[1].strip()
    except IndexError:
        # Manejo de error si el LLM no formatea bien la salida
        print("--- Error de formato en la decisión. Intentando de nuevo. ---")
        # Incrementa iteración y vuelve a razonar
        return {"iterations": state["iterations"] + 1} 

    # 2. ACTUAR
    result = llm_tool(query)
    
    # 3. OBSERVAR
    # Devolvemos el resultado como una "OBSERVACION".
    # Esto se añadirá al historial (gracias a operator.add)
    # y el cerebro lo leerá en el siguiente ciclo de razonamiento.
    return {
        "messages": [f"OBSERVACION: {result}"],
        "iterations": state["iterations"] + 1
    }

# Nodo 3: La "Voz" (Respuesta Final)
def generate_answer_node(state: AgentState):
    """
    Nodo final. Se ejecuta cuando el cerebro decide "ACCION: end".
    Sintetiza todo el historial (Pregunta, Pensamientos, Acciones y Observaciones)
    en una respuesta limpia y final para el usuario.
    """
    print("\n=== Nodo de Respuesta Final ===")
    history = "\n".join(state["messages"])
    
    # Prompt de síntesis: le pedimos al LLM que "limpie" el historial
    # y escriba una respuesta final coherente.
    prompt = f"""
Basado en el siguiente historial de conversación (pensamientos, acciones y observaciones), 
por favor, redacta una respuesta final completa, clara y bien estructurada para la pregunta inicial del usuario.

Historial:
{history}

Respuesta final para el usuario:
"""
    
    ### INICIO DE LA CORRECCIÓN ###
    # El error estaba aquí. El 'prompt' se invoca ANTES de ser
    # asignado a 'final_response'. Ahora es correcto.
    final_response = llm.invoke(prompt)
    # El último mensaje que se añade al estado es la respuesta final limpia
    return {"messages": [final_response.content.strip()]}
    ### FIN DE LA CORRECCIÓN ###


# --- PASO 5: Construcción del Grafo (Ensamblaje) ---
print("Construyendo el grafo del agente...")

# 1. Inicializamos el constructor del grafo, pasándole nuestro Estado
workflow = StateGraph(AgentState)

# 2. Añadimos los nodos que definimos (dándoles un nombre)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("action", action_node)
workflow.add_node("generate_answer", generate_answer_node)

# 3. Definimos el punto de entrada del grafo
workflow.set_entry_point("reasoning") # El agente siempre empieza "pensando"

# 4. Definimos el "Router" o Arista Condicional
def should_continue(state: AgentState):
    """
    Esta función es el "router" principal. Lee el campo 'next_action'
    del estado (que fue fijado por el nodo 'reasoning') y decide
    a qué nodo saltar a continuación.
    """
    if state.get("next_action") == "end":
        return "end" # Ir al nodo de generación de respuesta
    else:
        return "continue" # Ir al nodo de acción

# 5. Añadimos la Arista Condicional
# Desde el nodo 'reasoning', llamamos a la función 'should_continue'.
workflow.add_conditional_edges(
    "reasoning",
    should_continue,
    {
        "continue": "action",
        "end": "generate_answer"
    }
)

# 6. Añadimos la Arista del Bucle
# Esta es la arista que crea el BUCLE ReAct (Razonar -> Actuar -> Observar -> Razonar...)
workflow.add_edge("action", "reasoning")

# 7. Añadimos la Arista Final de Salida
# Después de que el nodo 'generate_answer' termina, el grafo va a 'END'.
workflow.add_edge("generate_answer", END)

# 8. Compilamos el grafo
app = workflow.compile()
print("Grafo compilado. ¡Listo para ejecutar!")

# --- PASO 6: Ejecución del Agente ---
if __name__ == "__main__":
    # La pregunta "meta" que le hacemos al agente
    initial_question = "Explica qué es el patrón ReAct en agentes de IA y por qué es útil, dame un ejemplo práctico."
    
    # Definimos el estado inicial con el que el grafo comenzará
    initial_state = {
        "messages": [f"Pregunta del usuario: {initial_question}"],
        "iterations": 0,
        "next_action": ""
    }
    
    print(f"\n\n--- Iniciando agente con la pregunta: '{initial_question}' ---")
    
    final_state = None
    # Usamos 'stream' para ver cada paso (cada nodo) del grafo a medida que ocurre
    for s in app.stream(initial_state, {"recursion_limit": 15}):
        print(s)
        print("---")
        final_state = s # Guardamos el último estado para imprimir la respuesta final

    # Imprimimos solo la respuesta final y limpia
    print("\n\n--- Ejecución Finalizada ---")
    print("Respuesta Final del Agente:")
    if final_state:
        # Extraemos el último mensaje (la respuesta limpia) del último estado
        final_message = next(iter(final_state.values()))['messages'][-1]
        print(final_message)
    else:
        print("No se pudo obtener el estado final.")
