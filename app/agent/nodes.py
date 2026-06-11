from pyexpat import model
from langchain_core.messages import SystemMessage
import logging 

logger = logging.getLogger(__name__)

# Template to create a node 
def create_agent_node(model_with_tools, system_prompt):
    def agent_node(state):
        logger.info("AGENT NODE: Processing request...")

        system_msg = SystemMessage(content=system_prompt)
        
        # Combine the system message with the conversation history stored in state
        # System message goes first so that model knows how to behave
        messages = [system_msg] + list(state["messages"])
        
        # model can decide to execute tools
        response = model_with_tools.invoke(messages)

        return {"messages": [response]}
    
    return agent_node

# wraps logging behavior to existing tool
def create_tool_node_with_logging(original_tool_node):
    def tool_node_with_logging(state):
        logger.info("TOOL NODE: Executing tools...")
        result = original_tool_node.invoke(state)
        logger.info("Tools executed successfully")
        return result
    return tool_node_with_logging

