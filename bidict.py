class BiDict():
    def __init__(self):
        self.fwd = {}
        self.rev = {}

    def insert(self, v1, v2):
        self.fwd[v1] = v2
        self.rev[v2] = v1

    def hasFirst(self, v1):
        return v1 in self.fwd

    def hasSecond(self, v2):
        return v2 in self.rev

    def getFirst(self, v1):
        if v1 in self.fwd:
            return self.fwd[v1]
        else:
            return None

    def getSecond(self, v2):
        if v2 in self.rev:
            return self.rev[v2]
        else:
            return None

    def __repr__(self):
        out = ""
        for key, val in self.fwd.items():
            out += str(key) + " -> " + str(val) + ", "
        return out