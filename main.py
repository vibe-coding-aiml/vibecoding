import os
from pathlib import Path
from IPython.display import Image, display # type: ignore
import autogen # type: ignore
from autogen.coding import LocalCommandLineCodeExecutor # type: ignore
from project_memory import ProjectMemory, StructureManager
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
# Limit the message history to the 3 most recent messages
max_msg_transfrom = transforms.MessageHistoryLimiter(max_messages=3)

# Limit the token limit per message to 10 tokens
token_limit_transform = transforms.MessageTokenLimiter(max_tokens_per_message=5900, min_tokens=10)
llm_config = {
    "model": "llama3-70b-8192",  # Or "mixtral-8x7b-32768"
    "base_url": "https://api.groq.com/",  # Groq's API base
    "api_key": "",
    "api_type": "groq", 
    "cache_seed": 41,  # seed for caching and reproducibility
    "temperature": 0, 
}

base_dir = Path(os.getcwd())  / 'webapp'
code_runner = LocalCommandLineCodeExecutor(work_dir=base_dir)

if not os.path.exists(base_dir):
    print(f"Path '{base_dir}' does not exist. Aborting.")
    exit(1)

memory = ProjectMemory(base_dir)

class MemoryAwareUserProxyAgent(autogen.UserProxyAgent):
    def __init__(self, *args, memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory

    def _execute_code_block(self, code, language):
        result = super()._execute_code_block(code, language)
        if self.memory:
            self.memory.update()
        return result

structure_manager = StructureManager(base_dir)

print(structure_manager.get_structure())

data_scientist = autogen.AssistantAgent(
        name="DataScientist",
        llm_config=llm_config,
        system_message=(
            "You are a data scientist leading a machine learning project. For each step (loading → EDA → preprocessing → modeling → tuning), " \
            f"Current Project structure: {structure_manager.get_structure()}. Instruct the code generator agent with the structure information, where to create, fetch the folder and files."
        ),
        # code_execution_config={"executor": code_runner,}
    )

code_generator_agent = autogen.UserProxyAgent(
    name="Code Generator Agent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=100,
    system_message=(
            "Main Goals: 1. Always reply what you've done, 2. Execute the generated code and get suggestions from data scientist"
            "You are a senior ML engineer building a machine learning project."
            f" Your workspace is `{base_dir}`."
            "\n\nProject Structure:\n"
            f"{structure_manager.get_structure()}\n"
            "Create a file in a respective folder if required."
            "\nYou are allowed to create folders and files like `pipeline`, `data`, `models`, `notebooks`, etc., under this root."
            " Always use absolute paths based on this root directory. You are responsible for creating and organizing files."
            "\nIf the folder doesn't exist, create it before writing files."
            "\nUse `open()` in write mode or `os.makedirs()` to create directories as needed."
            "\nTo test functionality, you can run scripts using subprocess or import them directly."
            "\nIf you run into module not found error, install the package using pip"
        ),
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        # the executor to run the generated code
        "executor": code_runner,
    },
)

# ADD NEW AGENT
file_manager_agent = autogen.AssistantAgent(
    name="FileManagerAgent",
    llm_config=llm_config,
    system_message=(
        "You are responsible for managing files and directories under the project root."
        f" Root path: `{base_dir}`."
        " Create folders/files when instructed. Use full paths and ensure structure is maintained."
    )
)

# suppose this capability is not available
context_handling = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
        transforms.MessageTokenLimiter(max_tokens=6000, max_tokens_per_message=500, min_tokens=50),
    ]
)
# INCLUDE NEW AGENT IN CONTEXT HANDLING
context_handling.add_to_agent(file_manager_agent)
context_handling.add_to_agent(data_scientist)
context_handling.add_to_agent(code_generator_agent)

# REDEFINE GROUP CHAT WITH NEW AGENT
group_chat = autogen.GroupChat(
    agents=[data_scientist, code_generator_agent, file_manager_agent],
    messages=[],
    max_round=50,
    speaker_selection_method="auto"
)

# RECREATE MANAGER
manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)





user_proxy = MemoryAwareUserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    memory=memory,
    code_execution_config={
        # the executor to run the generated code
        "executor": code_runner,
    },
)

chat_res = user_proxy.initiate_chat(
    manager,
    message="""Use the titanic.csv file to create a best model (Preferance is your choice).""",
    summary_method="reflection_with_llm",
)
