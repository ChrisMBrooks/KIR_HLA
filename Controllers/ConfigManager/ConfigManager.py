import json
import os

class ConfigManaager():
    def __init__(self):
        self.setup()

    def setup(self):
        full_path = os.path.join(os.getcwd(), "config.json")
        f = open(full_path)
        self.config = json.load(f)