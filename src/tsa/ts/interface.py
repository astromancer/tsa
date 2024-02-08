
from recipes.logging import LoggingMixin


class Interface(LoggingMixin):

    def __get__(self, instance, kls):
        if instance:  # lookup from instance
            self.parent = instance

        return self  # lookup from class

    def get_data(self, data):
        if data:
            return data

        if self.parent is not None:
            return tuple(self.parent)

        raise ValueError(f'Please provide data for {type(self).__name__}.')
