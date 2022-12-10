from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self.report = ""
    
    @abstractmethod
    def determine_run(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, conn):
        raise NotImplementedError