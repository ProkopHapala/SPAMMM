import os, pytest, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

@pytest.fixture
def xyz():
    return lambda n: os.path.join(DATA, 'xyz', n)

@pytest.fixture
def substrate():
    return lambda n: os.path.join(DATA, 'substrates', n)

@pytest.fixture
def dat():
    return lambda n: os.path.join(DATA, n)
