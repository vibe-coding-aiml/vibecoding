# memory/project_memory.py

import os

class ProjectMemory:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.folders = []
        self.files = []
        self.update()

    def update(self):
        """Scan and refresh the project folder and file structure."""
        self.folders.clear()
        self.files.clear()
        for root, dirs, filenames in os.walk(self.base_dir):
            for d in dirs:
                rel_dir = os.path.relpath(os.path.join(root, d), self.base_dir)
                self.folders.append(rel_dir)
            for f in filenames:
                rel_file = os.path.relpath(os.path.join(root, f), self.base_dir)
                self.files.append(rel_file)

    def get_structure_str(self):
        return (
            f"Base Directory: {self.base_dir}\n"
            f"Folders: {self.folders}\n"
            f"Files: {self.files}\n"
        )

    def file_exists(self, rel_path):
        return rel_path in self.files

    def folder_exists(self, rel_path):
        return rel_path in self.folders

class StructureManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.memory = ProjectMemory(base_dir)

    def get_structure(self):
        return self.memory.get_structure_str()

    def update(self):
        self.memory.update()