from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def generate_report(self, data_ref, data_prod):
        pass
