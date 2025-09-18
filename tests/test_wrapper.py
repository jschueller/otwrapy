import otwrapy.examples.beam as ex_beam
import otwrapy as otw
import openturns as ot
import multiprocessing
import pytest
import importlib

try:
    importlib.import_module("pathos")
    have_pathos = True
except ImportError:
    have_pathos = False

try:
    importlib.import_module("pydefx")
    have_pydefx = True
except ImportError:
    have_pathos = False

try:
    importlib.import_module("ipyparallel")
    have_ipyparallel = True
except ImportError:
    have_ipyparallel = False

try:
    importlib.import_module("joblib")
    have_joblib = True
except ImportError:
    have_joblib = False


def backendtest(backend):
    n_cpu = multiprocessing.cpu_count()
    sizes = [1, n_cpu, n_cpu + 1]
    if n_cpu > 2:
        sizes.append(n_cpu - 1)
    if n_cpu > 3:
        sizes.append(n_cpu - 2)
    model = ex_beam.Wrapper(sleep=0.2)
    if backend == "pydefx": # FIXME: cannot load python function from inside container
        model = ot.SymbolicFunction(["E", "F", "L", "I"], ["F * L^3 / (3.0 * E * I)"])
    dask_args = None
    ipp_client_kw = {}
    if backend == "dask":
        dask_args = {"scheduler": "localhost", "workers": {"localhost": n_cpu}}
    elif backend == "ipyparallel":
        import ipyparallel as ipp
        cluster = ipp.Cluster()
        cluster.start_controller_sync()
        cluster.start_engines_sync(n=n_cpu)
        ipp_client_kw = {"cluster_id": cluster.cluster_id}
    model_parallel = otw.Parallelizer(model, backend=backend, n_cpus=n_cpu,
                                      dask_args=dask_args, ipp_client_kw=ipp_client_kw)
    for size in sizes:
        X_sample = ex_beam.distribution.getSample(size)
        Y_ref = model(X_sample)
        Y_sample = ot.Sample(model_parallel(X_sample))
        assert Y_ref == Y_sample, "samples do not match"


def test_serial():
    backendtest("serial")


def test_multiprocessing():
    backendtest("multiprocessing")


@pytest.mark.skipif(not have_joblib, reason="N/A")
def test_joblib():
    backendtest("joblib")


@pytest.mark.skipif(not have_ipyparallel, reason="N/A")
def test_ipython():
    backendtest("ipyparallel")


@pytest.mark.skipif(not have_pathos, reason="N/A")
def test_pathos():
    backendtest("pathos")


def test_concurrent_thread():
    backendtest("concurrent/thread")


def test_concurrent_process():
    backendtest("concurrent/process")


@pytest.mark.skipif(not have_pydefx, reason="N/A")
def test_pydefx():
    backendtest("pydefx")


@pytest.mark.skip(reason="needs passwordless ssh configuration")
def test_dask_ssh():
    backendtest("dask/ssh")


@pytest.mark.skip(reason="needs a SLURM cluster")
def test_dask_slurm():
    backendtest("dask/slurm")
