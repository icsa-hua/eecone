from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def generate_streamlit_report(self, data_ref, data_prod, drift_method_placeholder):
        pass
