from numpy.random import Generator, Philox


class CounterGenerator:
    def __init__(self, seed):
        self.key = seed.generate_state(n_words=2, dtype='uint64')

    def get_generator(self, counter=0):
        return Generator(Philox(counter=counter, key=self.key))