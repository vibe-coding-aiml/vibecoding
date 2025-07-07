
import os
from pathlib import Path
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen import UserProxyAgent, GroupChat, GroupChatManager
from agents.code_writer_agent import create_code_writer
from agents.evaluator_agent import create_evaluator
# from agents.debug_agent import create_debugger
from agents.data_scientist_agent import create_data_scientist
from memory.project_memory import ProjectMemory, StructureManager
from utils.code_utils import extract_code_blocks_from_message
from utils.file_writer import write_code_to_file
from dotenv import load_dotenv

load_dotenv()

# Track ML pipeline stage
pipeline_stage = {
    "data_loaded": False,
    "eda_done": False,
    "transformation_done": False,
    "model_trained": False,
    "evaluation_done": False,
    "finalized": False
}
# STEP 1: Ask for the base directory
# base_dir = input("Enter Django base directory (e.g., webapp): ").strip()
base_dir = Path(os.getcwd()) / "webapp"
print(str(base_dir))

if not os.path.exists(base_dir):
    print(f"Path '{base_dir}' does not exist. Aborting.")
    exit(1)

# STEP 2: Initialize shared memory
memory = ProjectMemory(base_dir)

# STEP 3: Create memory-aware user proxy agent
class MemoryAwareUserProxyAgent(UserProxyAgent):
    def __init__(self, *args, memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory

    def _execute_code_block(self, code, language):
        result = super()._execute_code_block(code, language)
        if self.memory:
            self.memory.update()
        return result

# STEP 4: Inject shared memory and roles
# structure_str = memory.get_structure_str()
structure_manager = StructureManager(base_dir)

code_writer = create_code_writer(structure_manager)
evaluator = create_evaluator(structure_manager)
# debugger = create_debugger(structure_manager)
data_scientist = create_data_scientist()


# STEP 5: Define user proxy agent
user_proxy = MemoryAwareUserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    memory=memory,
    code_execution_config={"use_docker": False}
)

# STEP 6: Group chat
group_chat = GroupChat(
    agents=[user_proxy, code_writer, evaluator, data_scientist],
    messages=[],
    max_round=50
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "llama3-8b-8192", 
                # "base_url": "https://api.groq.com/openai/v1", 
                "base_url": "https://api.groq.com/", 
                "api_key": os.getenv('GROQ_API_KEY'),
                "api_type": "groq",
                # "price": [0.0, 0.0]
                }
)

def extract_and_save(name, filename):
    msg = next((m["content"] for m in reversed(group_chat.messages)
               if m["role"] == "assistant" and m["name"] == name), None)
    if msg:
        code_blocks = extract_code_blocks_from_message(msg, language="python")
        if code_blocks:
            file_path = os.path.join(memory.base_dir, filename)
            write_code_to_file(code_blocks[0], file_path)
            memory.update()
            print(f"Code saved to {file_path}\n")
        else:
            print(f"No Python code block found from {name}.")
    else:
        print(f"No message found from {name}.")

def ensure_data_folder(csv_path, base_dir):
    data_dir = os.path.join(base_dir, "pipeline", "data")
    os.makedirs(data_dir, exist_ok=True)
    dest_path = os.path.join(data_dir, os.path.basename(csv_path))
    shutil.copy(csv_path, dest_path)
    print(f"CSV copied to {dest_path}")
    return dest_path

# STEP 7: Run chat

def run_pipeline():
    print("\nStarting agentic ML orchestration...\n")

    # Ask for CSV file path
    csv_path = 'titanic.csv' #input("Enter the full path to your CSV file: ").strip()
    if not os.path.isfile(csv_path) or not csv_path.endswith(".csv"):
        print("Invalid file. Please provide a valid .csv path.")
        return

    final_csv_path = ensure_data_folder(csv_path, base_dir)
    memory.update()

    # Step 1: Write data loader script
    user_proxy.initiate_chat(manager, message=(
        f"Write a script called `data_loader.py` under `pipeline/` that defines a function `load_data(file_path)`."
        f" The dataset is located at: {final_csv_path}. It should load CSV data and return a pandas DataFrame."
    ))
    extract_and_save("CodeWriter", "pipeline/data_loader.py")
    pipeline_stage["data_loaded"] = True

    # Step 2: Evaluator runs the script
    user_proxy.initiate_chat(manager, "Please evaluate `data_loader.py` and confirm it works.")

    # Step 3: DataScientist analyzes structure & suggests model
    user_proxy.initiate_chat(manager, (
        "Please analyze the dataset loaded by `load_data()` and determine if it is a classification or regression task."
        " Recommend a few ML models to try."
    ))
    pipeline_stage["eda_done"] = True

    # Step 4: CodeWriter writes training script
    user_proxy.initiate_chat(manager, (
        "Based on the DataScientist's recommendation, write `train_model.py` in `pipeline/` that trains all suggested models,"
        " evaluates them using appropriate metrics, and prints a comparison of the results."
    ))
    extract_and_save("CodeWriter", "pipeline/train_model.py")
    pipeline_stage["model_trained"] = True

    # Step 5: Evaluator runs training
    user_proxy.initiate_chat(manager, "Run `train_model.py` and report the performance of each model.")
    pipeline_stage["evaluation_done"] = True

    # Step 6: DataScientist final recommendation
    user_proxy.initiate_chat(manager, (
        "Based on the performance metrics, please recommend the best model and suggest further improvements if needed."
    ))
    pipeline_stage["finalized"] = True

if __name__ == "__main__":
    run_pipeline()
