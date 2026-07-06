import os, pytest, numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def pytest_addoption(parser):
    parser.addoption('--update-refs', action='store_true', default=False,
                     help='Update reference data files instead of comparing')
    parser.addoption('--visual', action='store_true', default=False,
                     help='Generate PNG visual artifacts (L2 human review)')
    parser.addoption('--review', action='store_true', default=False,
                     help='Write .out/.log agent-review artifacts (L1)')
    parser.addoption('--develop', action='store_true', default=False,
                     help='Develop mode: --review + --visual + high verbosity')


def pytest_configure(config):
    if config.getoption('--develop', default=False):
        import spammm.globals as g
        g.set_develop_mode(True)


def pytest_ignore_collect(collection_path, config):
    return collection_path.name.startswith('testplot_')


def _develop(request):
    return request.config.getoption('--develop', default=False)


def _visual_on(request):
    return request.config.getoption('--visual', default=False) or _develop(request)


def _review_on(request):
    return request.config.getoption('--review', default=False) or _develop(request)


def _module_debug_dir(request):
    module_file = request.module.__file__
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    outdir = os.path.join('debug', module_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture
def update_refs(request):
    return request.config.getoption('--update-refs')


@pytest.fixture
def develop_mode(request):
    return _develop(request)


@pytest.fixture
def review_enabled(request):
    return _review_on(request)


@pytest.fixture
def visual_output_dir(request):
    """Returns debug/<script>/ if --visual or --develop, else None."""
    if not _visual_on(request):
        return None
    return _module_debug_dir(request)


@pytest.fixture
def review_dir(request):
    """Base output dir debug/<script>/ when --review or --develop."""
    if not _review_on(request):
        return None
    return _module_debug_dir(request)


@pytest.fixture
def make_review(request, review_dir):
    """Factory: make_review('test_foo') -> ReviewSession. No-op when review off."""
    from tests.helpers.review import ReviewSession

    def _make(test_name: str):
        enabled = _review_on(request)
        outdir = review_dir or os.path.join('debug', 'unknown')
        return ReviewSession(outdir, test_name, enabled=enabled)

    return _make


@pytest.fixture
def xyz():
    return lambda n: os.path.join(DATA, 'xyz', n)


@pytest.fixture
def substrate():
    return lambda n: os.path.join(DATA, 'substrates', n)


@pytest.fixture
def dat():
    return lambda n: os.path.join(DATA, n)
