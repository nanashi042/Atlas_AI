class MemoryService:

    def __init__(self):
        self.memory = {}

    def save(self, user_id: int, role: str, message: str):
        if user_id not in self.memory:
            self.memory[user_id] = []

        self.memory[user_id].append(
            {
                "role": role,
                "message": message,
            }
        )

    def get(self, user_id: int):
        return self.memory.get(user_id, [])