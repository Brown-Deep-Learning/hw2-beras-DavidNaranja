import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.
        
        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        labels = np.unique(data)
        s = len(labels)
        one_hots = np.eye(s,s)
        
        self.one_dict = {labels[i]:one_hots[i, :] for i in range(s)}
        self.inv_dict = {i:labels[i] for i in range(s)}

    def forward(self, data):
        try:
            self.one_dict
        except AttributeError:
            self.fit(data)
        return np.array([self.one_dict[imp] for imp in data])

    def inverse(self, data):
        try:
            self.inv_dict
        except AttributeError:
            self.fit(data)
        argm = np.argmax(data, -1)
        return np.array([[self.inv_dict[m] for m in argm]])
