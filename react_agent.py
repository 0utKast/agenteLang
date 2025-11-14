import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- Configuración Inicial ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

# Se especifica un modelo de la lista de disponibles para el usuario.
llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

# --- Definición del Estado del Agente ---
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_action: str
    iterations: int

# --- Herramientas del Agente ---
def llm_tool(query: str) -> str:
    """Usa el LLM principal para responder una consulta específica."""
    print(f"--- Usando herramienta LLM para la consulta: {query} ---")
    response = llm.invoke(f"Responde esta pregunta de forma concisa: {query}")
    return response.content.strip()

# --- Nodos del Grafo ---

def reasoning_node(state: AgentState):
    """El "cerebro" del agente. Decide qué hacer a continuación."""
    print("\n=== Nodo de Razonamiento ===")
    
    if state["iterations"] >= 5:
        print("--- Límite de iteraciones alcanzado. Forzando finalización. ---")
        return {"next_action": "end"}

    history = "\n".join(state["messages"])
    
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
    
    response = llm.invoke(prompt)
    decision = response.content.strip()
    print(f"Decisión del LLM: {decision}")

    if "ACCION: end" in decision:
        return {"messages": [decision], "next_action": "end"}
    else:
        return {"messages": [decision], "next_action": "action"}

def action_node(state: AgentState):
    """El nodo de "acción". Ejecuta la herramienta."""
    print("\n=== Nodo de Acción ===")
    last_thought = state["messages"][-1]
    
    try:
        query = last_thought.split("ACCION:")[1].strip()
    except IndexError:
        print("--- Error de formato en la decisión. Intentando de nuevo. ---")
        return {"iterations": state["iterations"] + 1}

    result = llm_tool(query)
    
    return {
        "messages": [f"OBSERVACION: {result}"],
        "iterations": state["iterations"] + 1
    }

def generate_answer_node(state: AgentState):
    """Nodo final: Genera la respuesta completa para el usuario."""
    print("\n=== Nodo de Respuesta Final ===")
    history = "\n".join(state["messages"])
    
    prompt = f"""
Basado en el siguiente historial de conversación (pensamientos, acciones y observaciones), 
por favor, redacta una respuesta final completa, clara y bien estructurada para la pregunta inicial del usuario.

Historial:
{history}

Respuesta final para el usuario:
"""
    
    final_response = llm.invoke(prompt)
    return {"messages": [final_response.content.strip()]}


# --- Construcción del Grafo ---
print("Construyendo el grafo del agente...")
workflow = StateGraph(AgentState)

workflow.add_node("reasoning", reasoning_node)
workflow.add_node("action", action_node)
workflow.add_node("generate_answer", generate_answer_node)

workflow.set_entry_point("reasoning")

def should_continue(state: AgentState):
    """Router: Decide si continuar al nodo de acción o finalizar."""
    if state.get("next_action") == "end":
        return "end"
    else:
        return "continue"

workflow.add_conditional_edges(
    "reasoning",
    should_continue,
    {
        "continue": "action",
        "end": "generate_answer"  # En lugar de END, vamos al nuevo nodo
    }
)
workflow.add_edge("action", "reasoning")
workflow.add_edge("generate_answer", END) # El nuevo nodo finaliza el grafo

app = workflow.compile()
print("Grafo compilado. ¡Listo para ejecutar!")

# --- Ejecución del Agente ---
if __name__ == "__main__":
    initial_question = "Explica qué es el patrón ReAct en agentes de IA y por qué es útil, dame un ejemplo práctico."
    
    initial_state = {
        "messages": [f"Pregunta del usuario: {initial_question}"],
        "iterations": 0,
        "next_action": ""
    }
    
    print(f"\n\n--- Iniciando agente con la pregunta: '{initial_question}' ---")
    
    final_state = None
    for s in app.stream(initial_state, {"recursion_limit": 15}):
        print(s)
        print("---")
        final_state = s

    print("\n\n--- Ejecución Finalizada ---")
    print("Respuesta Final del Agente:")
    if final_state:
        final_message = next(iter(final_state.values()))['messages'][-1]
        print(final_message)
    else:
        print("No se pudo obtener el estado final.")
