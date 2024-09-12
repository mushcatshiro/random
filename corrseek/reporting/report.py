import os


class ReportGenerator:
    def __init__(self, name, storage=None):
        self.storage = self._storage(name, storage)
    
    def _storage(self, name, storage):
        if storage:
            if os.path.exists(storage):
                return os.path.join(storage, name)
            else:
                raise FileNotFoundError(
                    f"storage directory {storage} not found"
                )
        else:
            return os.path.join(os.getcwd(), name)

    def run(self):
        tmpl = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "reporttmpl.html"
        )
        pass