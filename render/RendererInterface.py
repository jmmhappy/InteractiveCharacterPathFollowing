from abc import *

class RendererInterface(metaclass=ABCMeta):
    def __init__(self, frametime):
        # Need to initiate this constant
        self.FRAMETIME = frametime

    @abstractmethod
    def getFocusPoint(self):
        return 
    @abstractmethod
    def render(self, shadow, options): # Draw
        return
    @abstractmethod
    def update(self, userInput): # Timer
        return
    @abstractmethod
    def reset(self):
        return
