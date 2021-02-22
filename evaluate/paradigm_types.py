from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Form:
    """A single surface form from a paradigm"""
    lemma: str = None
    word: str = None
    msd: str = None

    def __eq__(self, other):
        """Evaluation ignores lemma and msd, if provided, so we compute quality
        only wrt the surface word"""
        return self.word == other.word

    @classmethod
    def from_line(cls, line: str):
        line = line.strip()

        parts = line.split("\t")

        if len(parts) == 3:
            l, w, m = parts
            return cls(lemma=l, word=w, msd=m)
        elif len(parts) == 2:
            l, w = parts
            return cls(lemma=l, word=w)
        elif len(parts) == 1:
            w = parts[0]
            return cls(word=w)
        else:
            raise Exception(f"Unable to parse {line} into a Paradigm Form")


@dataclass
class Paradigm:
    lemma: str = None
    forms: List[Form] = field(default_factory=list)
    _words: List[str] = None

    def __len__(self):
        return len(self.forms)

    @property
    def words(self):
        if self._words is None:
            self._words = [f.word for f in self.forms]

        return self._words
