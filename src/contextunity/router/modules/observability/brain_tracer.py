from langchain_core.callbacks import AsyncCallbackHandler


class BrainTraceCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        super().__init__()
        self.steps = []
        self._spans = {}

    # Implement callbacks...
