"""Activation implementations"""

from torch.nn import LeakyReLU, Module, ReLU, Sequential

from model.interface import IActivationFactory


class EmptyActivationFactory(IActivationFactory):
    """Factory for generating placeholder normalization layers"""

    def build(self) -> Module:
        return Sequential()


class LeakyReLUFactory(IActivationFactory):
    """Factory for generating leaky relu activations"""

    def __init__(self, negative_slope: float) -> None:
        self._negative_slope = negative_slope

    def build(self) -> Module:
        return LeakyReLU(negative_slope=self._negative_slope)


class ReLUFactory(IActivationFactory):
    """Factory for generating relu activations"""

    def build(self) -> Module:
        return ReLU()


def get_activation_factory(activation_factory: IActivationFactory | None) -> IActivationFactory:
    """Conver to empty activation factory if None"""
    if activation_factory is None:
        return EmptyActivationFactory()
    return activation_factory
