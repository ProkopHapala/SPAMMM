import os, pytest, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def pytest_addoption(parser):
    parser.addoption('--update-refs', action='store_true', default=False,
                     help='Update reference data files instead of comparing')

@pytest.fixture
def update_refs(request):
    return request.config.getoption('--update-refs')

@pytest.fixture
def xyz():
    return lambda n: os.path.join(DATA, 'xyz', n)

@pytest.fixture
def substrate():
    return lambda n: os.path.join(DATA, 'substrates', n)

@pytest.fixture
def dat():
    return lambda n: os.path.join(DATA, n)
