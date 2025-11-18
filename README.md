# Proyecto 1: Agente con LangChain y LangGraph

Este es el primer proyecto de una serie de experimentos y aplicaciones construidas durante un curso enfocado en LangChain y LangGraph.

## Descripción

Este proyecto inicial sienta las bases para la construcción de agentes inteligentes utilizando las librerías de LangChain. El objetivo es explorar los conceptos fundamentales de cadenas (Chains), grafos (Graphs) y la creación de agentes reactivos.

## Contenidos del Repositorio

*   `cadena_langchain.py`: Implementación de una cadena simple con LangChain.
*   `cadena_langgraph.py`: Evolución de la cadena simple a un grafo con LangGraph para manejar flujos más complejos.
*   `react_agent.py`: Implementación detallada de un agente reactivo (ReAct) utilizando LangGraph, con explicaciones paso a paso del patrón ReAct, sus nodos y el flujo de ejecución.
*   `.env`: Archivo de ejemplo para la configuración de variables de entorno (API keys, etc.). **Nota: Este archivo no se sube al repositorio.**
*   `.gitignore`: Archivo para especificar los archivos y directorios que Git debe ignorar.

## Patrón ReAct (Reasoning and Acting)

El patrón ReAct es una técnica que permite a los Large Language Models (LLMs) combinar el razonamiento (Reasoning) con la actuación (Acting) para resolver tareas complejas. En esencia, el LLM genera pensamientos (PENSAMIENTO) para planificar y reflexionar sobre la tarea, y luego decide qué acción (ACCION) tomar, que generalmente implica el uso de herramientas. Después de ejecutar la acción, el LLM observa el resultado (OBSERVACION) y lo utiliza para su siguiente ciclo de razonamiento.

`react_agent.py` es un ejemplo práctico de cómo implementar este patrón utilizando LangGraph, estructurando el agente en nodos de razonamiento, acción y generación de respuesta final.