import os, pytest, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def pytest_addoption(parser):
    parser.addoption('--update-refs', action='store_true', default=False,
                     help='Update reference data files instead of comparing')
    parser.addoption('--visual', action='store_true', default=False,
                     help='Generate visual output images (PNG) for human review')

@pytest.fixture
def update_refs(request):
    return request.config.getoption('--update-refs')

@pytest.fixture
def visual_output_dir(request):
    """Returns output dir path if --visual flag set, else None.
    Images go to debug/<script_filename>/ following the output location policy.
    """
    if request.config.getoption('--visual', default=False):
        module_file = request.module.__file__
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        outdir = os.path.join('debug', module_name)
        os.makedirs(outdir, exist_ok=True)
        return outdir
    return None

@pytest.fixture
def xyz():
    return lambda n: os.path.join(DATA, 'xyz', n)

@pytest.fixture
def substrate():
    return lambda n: os.path.join(DATA, 'substrates', n)

@pytest.fixture
def dat():
    return lambda n: os.path.join(DATA, n)
