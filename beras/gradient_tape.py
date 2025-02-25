from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        #It's fair to assume the sources are always weights and biases
        while len(queue) > 0:
            
            #1. Take out first in queue and get its associated previous layer
            current_id = id(queue.pop(0))
            current_layer = self.previous_layers[current_id]
            if current_layer is None:
                continue
            
            forwards_j = grads[current_id]
            

            #2. Update queue to have the previous layer's inputs, assuming they have an associated layer
            daughters = current_layer.inputs
            w_daughters = current_layer.weights
            queue += daughters

            #3. Compute overall grads and update dicts
            cumulative_grads = current_layer.compose_input_gradients(forwards_j)
            for d, g in zip(daughters, cumulative_grads):
                grads[id(d)] = [g]
            
            cumulative_grads = current_layer.compose_weight_gradients(forwards_j)
            if w_daughters is not None and len(w_daughters) > 0:
                for w, g in zip(w_daughters, cumulative_grads):
                    grads[id(w)] = [g]
        
        print(sources)
        return [Tensor(grads[id(s)][0]) for s in sources]

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

'''
def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        #It's fair to assume the sources are always weights and biases
        while len(queue) > 0:
            
            #1. Take out first in queue and get its associated previous layer
            q_id = id(queue.pop(0))

            current_layer = self.previous_layers[q_id]
            if current_layer is None:
                continue
            forwards_j = grads[q_id]

            #2. Update queue to have the previous layer's inputs, assuming they have an associated layer
            daughters = current_layer.inputs
            w_daughters = current_layer.weights
            queue += [x for x in daughters if id(x) in self.previous_layers]

            #3. Compute overall grads and update dicts
            cumulative_grad = current_layer.compose_input_gradients(forwards_j)
            for d, g in zip(daughters, cumulative_grad):
                print(d)
                print(g)
                grads[id(d)] = g

            if w_daughters is not None and len(w_daughters) != 0:
                wb_grads = current_layer.compose_weight_gradients(forwards_j)
                for w, g in zip(w_daughters, wb_grads):
                    grads[id(w)] = g

        #for k in grads:
         #   print(k)
          #  print(grads[k])
        return [Tensor(grads[id(s)]) for s in sources]

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        # '''