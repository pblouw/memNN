import numpy as np

from nengo import spa
from nengo.spa import pointer

from nengo.utils.compat import is_iterable, is_number, is_integer, range

class TextVocabulary(spa.Vocabulary):
    """
    Subclasses spa.Vocabulary to override constraints requiring semantic pointers to
    start with capital letters.
    """
    def __getitem__(self, key):
        """Return the semantic pointer with the requested name.

        If one does not exist, automatically create one.  The key must be
        a valid semantic pointer name, which is any Python identifier starting
        with a capital letter.
        """
        value = self.pointers.get(key, None)
        if value is None:
            if is_iterable(self.unitary):
                unitary = key in self.unitary
            else:
                unitary = self.unitary
            value = self.create_pointer(unitary=unitary)
            self.add(key, value)
        return value

    def add(self, key, p):
        """Add a new semantic pointer to the vocabulary.

        The pointer value can be a SemanticPointer or a vector.
        """
        if not isinstance(p, pointer.SemanticPointer):
            p = pointer.SemanticPointer(p)

        if key in self.pointers:
            raise KeyError("The semantic pointer '%s' already exists" % key)

        self.pointers[key] = p
        self.keys.append(key)
        self.vectors = np.vstack([self.vectors, p.v])

        # Generate vector pairs
        if self.include_pairs and len(self.keys) > 1:
            for k in self.keys[:-1]:
                self.key_pairs.append('%s*%s' % (k, key))
                v = (self.pointers[k] * p).v
                self.vector_pairs = np.vstack([self.vector_pairs, v])
