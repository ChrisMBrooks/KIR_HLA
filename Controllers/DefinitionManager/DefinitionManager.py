import json
import os

class DefinitionManaager():
    def __init__(self,filename:str):
        self.filename = filename
        self.setup()

    def setup(self):
        path = os.path.join(os.getcwd(), 'Definitions')
        full_path = os.path.join(path, self.filename)
        f = open(full_path)
        self.definition = json.load(f)