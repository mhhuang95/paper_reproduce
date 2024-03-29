import numpy as np

from abc import ABC, abstractmethod


class WrapperBase(ABC):
    def __init__(self, wrapped_layer):
        """Abstract base class for all Wrapper instances"""
        self._base_layer = wrapped_layer
        if hasattr(wrapped_layer, "_base_layer"):
            self._base_layer = wrapped_layer._base_layer
        super().__init__()

    @abstractmethod
    def _init_wrapper_params(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Overwirtten by inherited class"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Overwirtten by inherited class"""
        raise NotImplementedError

    @property
    def trainable(self):
        """Whether the base layer is frozen"""
        return self._base_layer.trainable

    @property
    def parameters(self):
        """A dictionary of the base layer parameters"""
        return self._base_layer.parameters

    @property
    def hyperparameters(self):
        """A dictionary of the base layer's hyperparameters"""
        hp = self._base_layer.hyperparameters
        hpw = self._wrapper_hyperparameters
        if "wrappers" in hp:
            hp["wrappers"].append(hpw)
        else:
            hp["wrappers"] = [hpw]
        return hp

    @property
    def derived_variables(self):
        """
        A dictionary of the intermediate values conputed during layer training
        """
        dv = self._base_layer.derived_variables.copy()
        if "wrappers" in dv:
            dv["wrappers"].append(self._wrapper_derived_variables)
        else:
            dv["wrappers"] = [self._wrapper_derived_variables]
        return dv

    @property
    def gradients(self):
        """A dictionary of the current layer parameter gradients"""
        return self._base_layer.gradients

    @property
    def act_fn(self):
        """The activation function for the base layer"""
        return self._base_layer.act_fn

    @property
    def X(self):
        """The collection of layer inputs"""
        return self._base_layer.X

    def _init_params(self):
        hp = self._wrapper_hyperparameters
        if "wrappers" in self._base_layer.hyperparameters:
            self._base_layer.hyperparameters["wrappers"].append(hp)
        else:
            self._base_layer.hyperparameters["wrappers"] = hp

    def freeze(self):
        """
        Freeze the base layer's params at their current value so that they can
        no longer be updated.
        """
        self._base_layer.freeze()

    def flush_gradients(self):
        """Erase all the wrapper and base layer's derived variables and gradients"""
        assert self.trainable, "Layer is frozen"
        self._base_layer.flush_gradients()

        for k, v in self._wrapper_derived_variables.items():
            self._wrapper_derived_variables[k] = []

    def update(self, lr):
        """
        Update the base layer's parameters using the accurate gradients and
        layer optimizer. Flush all gradients once the update is done
        """
        assert self.trainable, "Layer is frozen"
        self._base_layer.update(lr)
        self.flush_gradients()

    def _set_wrapper_params(self, pdict):
        for k, v in pdict.items():
            if k in self._wrapper_hyperparameters:
                self._wrapper_hyperparameters[k] = v
        return self

    def set_params(self, summary_dict):
        """
        Set the base layer parameters from a dictionary of values
        :param summary_dict: A dictionary of layer parameters and hyperparameters.
        If a required parameter or hyperparameter is not included, this method
        will use the value in the current layer
        :return: The newly-initialized layer.
        """
        return self._base_layer.set_params(summary_dict)

    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID"""
        return {
            "layer": self.hyperparameters["layer"],
            "layer_wrappers": [i["wrappers"] for i in self.hyperparameters["wrappers"]],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters
        }


class Dropout(WrapperBase):
    def __init__(self, wrapped_layer, p):
        """
        A dropout regularization wrapper

        During training, a dropout layer zeroes each element of the layer input
        with probability `p` and scales the activation by `1 / (1 - p)` (to reflect
        the fact that on average only `(1 - p) * N` units are active on any
        training pass). At test time, does not adjust elements of the input at
        all (ie., simply computes the identity function).

        :param wrapper_layer: the layer to apply dropout
        :param p: the dropout probability
        """
        super().__init__(wrapped_layer)
        self.p = p
        self._init_wrapper_params()
        self._init_params()

    def _init_wrapper_params(self):
        self._wrapper_derived_variables = {"dropout_mask": []}
        self._wrapper_hyperparameters = {"wrapper": "Dropout", "p": self.p}

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output with dropout for a single minibatch
        :param X:
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples.
        :param retain_derived:
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.
        :return:
            Layer output for each of the `n_ex` examples.
        """
        scaler, mask = 1.0, np.ones(X.shape).astype(bool)
        if self.trainable:
            scaler = 1.0 / (1.0 - self.p)
            mask = np.random.rand(*X.shape) >= self.p
            X = X * mask

        if retain_derived:
            self._wrapper_derived_variables["dropout_mask"].append(mask)

        return scaler * self._base_layer.forward(X, retain_derived)

    def backward(self, dLdy, retain_grads):
        """
        Backprop from the base layer's output to inputs
        :param dLdy:
            The gradient(s) of the loss wrt. the layer output(s).
        :param retain_grads:
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        :return:
            The gradient of the loss wrt. the layer input(s) `X`.
        """
        assert self.trainable, "Layer is frozen"
        dLdy *= 1.0 / (1.0 - self.p)
        return self._base_layer.backward(dLdy, retain_grads)


def init_wrappers(layer, wrappers_list):
    for wr in wrappers_list:
        if wr["wrapper"] == "Dropout":
            layer = Dropout(layer, 1)._set_wrapper_params(wr)
        else:
            raise NotImplementedError("{}".format(wr["wrapper"]))
    return layer
