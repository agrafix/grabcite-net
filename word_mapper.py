from bidict import BiDict

class WordMapper:
    def __init__(self, vec, getter):
        vals = [getter(o) for o in vec if o != None and getter(o) != None]
        self.samples = len(vals)
        self.map = BiDict()
        self.counterMap = {}
        ctr = 2 # 1 is unk, 0 is padding
        for v in vals:
            if type(v) is list:
                for w in v:
                    ctr = self.handleWord(w, ctr)
            else:
                ctr = self.handleWord(v, ctr)

    def handleWord(self, w, ctr):
        if w is None:
            return ctr

        wid = self.map.getFirst(w)
        if wid is None:
            wid = ctr
            self.map.insert(w, ctr)
            ctr += 1
        if wid in self.counterMap:
            self.counterMap[wid] += 1
        else:
            self.counterMap[wid] = 1

        return ctr

    def restrictTo(self, limit):
        pairs = []
        for key, value in self.counterMap.items():
            pairs.append((key, value))
        pairs.sort(key=lambda x: x[1], reverse=True)

        restricted = pairs[:limit]
        new_dict = BiDict()
        new_key = 2
        for key, _ in restricted:
            tok = self.map.getSecond(key)
            new_dict.insert(tok, new_key)
            new_key += 1
        self.map = new_dict

    def catSize(self):
        return len(self.map.fwd) + 2

    def toId(self, v):
        v = self.map.getFirst(v)
        if v == None:
            return 0
        else:
            return v

    def listToId(self, lst):
        return [self.toId(v) for v in lst]