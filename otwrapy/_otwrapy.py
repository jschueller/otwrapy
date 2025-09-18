"""
General purpose OpenTURNS python wrapper tools
"""

import os
import gzip
import pickle
from tempfile import mkdtemp
import shutil
from functools import wraps
import logging
import warnings
import openturns as ot
import numpy as np
from tqdm import tqdm

__all__ = ['load_array', 'dump_array', 'safemakedirs', 'create_logger',
           'FunctionDecorator', 'TempWorkDir', 'Parallelizer', 'Debug']


base_dir = os.path.dirname(__file__)


def load_array(filename, compressed=False):
    """Load a (possibly compressed) pickled array.

    Parameters
    ----------
    filename : str
        Path to the file to be loaded. If the extension is '.pklz', it considers
        that the file is compressed with *gzip*.
    compressed : bool
        Indicates if the file is compressed with gzip or not.
    """
    if compressed or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(filename, 'rb') as fh:
            return pickle.load(fh)


def dump_array(array, filename, compress=False):
    """Dump an array to a (possibly compressed) file.

    Parameters
    ----------
    array : array
        Array to be compressed. Typically a np.array or ot.Sample.
    filename : str
        Path where the file is dumped. If the extension is '.pklz', it considers
        that the file has to be compressed with *gzip*.
    compressed : bool
        Indicates if the file has to be compressed with gzip or not.
    """
    if compress or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)
    else:
        with open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)


def safemakedirs(folder):
    """Make a directory without raising an error if it exists.

    Parameters
    ----------
    folder : str
        Path of the folder to be created.
    """
    try:
        os.makedirs(folder, exist_ok=True)
    except FileExistsError:
        # can still raise even with exist_ok because of concurent processes
        pass


def create_logger(logfile, loglevel=None):
    """Create a logger with a FileHandler at the given loglevel.

    Parameters
    ----------
    logfile : str
        Filename for the logger FileHandler to be created.

    loglevel : logging level
        Threshold for the logger. Logging messages which are less severe than
        loglevel will be ignored. It defaults to logging.DEBUG.
    """
    if loglevel is None:
        loglevel = logging.DEBUG

    logger = logging.getLogger('logger_otwrapy')
    logger.setLevel(loglevel)

    # ----------------------------------------------------------
    # Create file handler which logs even DEBUG messages
    fh = logging.FileHandler(filename=logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a formatter for the file handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')

    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger


class Debug(object):

    """Decorator that catches exceptions inside a function and logs them.

    A decorator used to protect functions so that exceptions are logged to a
    file. It can either be instantiated with a Logger or with a filename for
    which a logger will be created with a FileHandler. It comes specially handy
    when you launch your codes in a non interactive environment (e.g., HPC
    cluster through submission scripts), given that Exceptions are captured
    and logged to a file.

    The great benefit of this implementation is that with a simple decorator
    you can protect the methods of your Wrapper class with a try/except
    structure. However, this might not be useful for a deeper debugging where
    you want to have access to the locals() of the place where the Exception
    jumped. If you bump into such a case, add a try/except structure that
    catches the Exception on the specific place. It is advised to use the
    decorator once you have developed the wrapper and that you are ready to
    launch your uncertainty studies.

    Parameters
    ----------
    logger : logging.Logger or str
        Either a Logger instance or a filename for the logger to be created.

    loglevel : logging level
        Threshold for the logger. Logging messages which are less severe than
        loglevel will be ignored. It defaults to logging.DEBUG.

    Examples
    --------
    To catch exceptions raised inside a function and log them to a file :

    >>> import otwrapy as otw
    >>> @otw.Debug('func.log')
    >>> def func(*args, **kwargs):
    >>>     pass


    """

    def __init__(self, logger, loglevel=None):
        if isinstance(logger, logging.Logger):
            self.logger = logger
            if loglevel is not None:
                self.logger.setLevel(loglevel)
        elif isinstance(logger, str):
            self.logger = create_logger(logger, loglevel=loglevel)

    def __call__(self, func):
        @wraps(func)
        def func_debugged(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(e, exc_info=True)
                raise e

        return func_debugged


class FunctionDecorator(object):

    """Convert an OpenTURNSPythonFunction into a Function.

    This class is intended to be used as a decorator.

    Parameters
    ----------
    enableCache : bool (Optional)
        If True, enable cache of the returned ot.Function

    Examples
    --------
    In order to always get an ot.Function when instantiating your
    wrapper, decorate it as follows:

    >>> import otwrapy as otw
    >>> import openturns as ot
    >>> @otw.FunctionDecorator(enableCache=True)
    >>> class Wrapper(ot.OpenTURNSPythonFunction):
    >>>     pass

    Note that a great disadvantage of this decorator is that your wrapper cannot
    be parallelized afterwards. Only use it if you don't plan to parallelize
    your wrapper or if the wrapper itself is parallelized already. However, if
    you plan to use :class:`Parallelizer`, there is no need to use this decorator !


    Notes
    -----
    I wanted this decorator to work also with Wrapper class, but it only works
    with ParallelWrapper for the moment. The problem is that, apparently,
    decorated classes are not picklable, and Wrapper instances must be picklable
    so that they can be easily distributed with `multiprocessing`

    References
    ----------
    http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
    http://www.artima.com/weblogs/viewpost.jsp?thread=240808
    http://stackoverflow.com/questions/30714485/why-does-a-decorated-class-looses-its-docstrings
    http://stackoverflow.com/questions/30711730/decorated-class-looses-acces-to-its-attributes
    """

    def __init__(self, enableCache=True, doc=None):
        self.enableCache = enableCache
        self.doc = doc

    def __call__(self, wrapper):
        @wraps(wrapper)
        def numericalmathfunction(*args, **kwargs):
            wrapper_instance = wrapper(*args, **kwargs)
            func = ot.Function(wrapper_instance)
            # Enable cache
            if self.enableCache:
                func = ot.MemoizeFunction(func)
                func.disableHistory()

            # Update __doc__ of the function
            if self.doc is None:
                # Inherit __doc__ from ParallelWrapper.
                func.__doc__ = wrapper.__doc__
            else:
                func.__doc__ = self.doc

            # Add the kwargs as attributes of the function for reference
            # purposes.
            func.__dict__.update(kwargs)
            func.__dict__.update(wrapper_instance.__dict__)
            return func
        # Keep the wrapper class as reference
        numericalmathfunction.__wrapper__ = wrapper
        return numericalmathfunction


class TempWorkDir(object):

    """Implement a context manager that creates a temporary working directory.

    Create a temporary working directory in `dir` preceded by
    `prefix` and clean up at the exit if necessary.

    Parameters
    ----------
    dir : str, optional
        Root path where the temporary working directory will be created. If None,
        it will default to the platform dependent temporary working directory
        Default = None

    prefix : str, optional
        String that preceeds the directory name.
        Default = 'run-'

    cleanup : bool, optional
        If True erase the directory and its children at the exit.
        Default = False

    transfer : list, optional
        List of files or folders to transfer to the temporary working directory

    chdir : bool, optional
        Whether to change directory to the temporary working directory.
        Default is False, should not be necessary and intended for backward compatibility only.

    Examples
    --------
    In the following example, everything that is executed inside the `with`
    environment will happen at a temporary working directory created at
    :file:`/tmp` with :file:`/run-` as a prefix. The created directory will be
    erased upon the exit of the `with` environment, even if an Exception is raised.

    >>> import otwrapy as otw
    >>> import subprocess
    >>> with otw.TempWorkDir('/tmp', prefix='run-', cleanup=True) as cwd:
    >>> ... print(cwd)
    >>> ... # [write input files]
    >>> ... subprocess.run(["myexe", "input.txt"], cwd=cwd) # doctest: +SKIP
    >>> ... # [read output files]
    """

    def __init__(self, dir=None, base_temp_work_dir=None, prefix='run-',
                 cleanup=False, transfer=None, chdir=False):
        if base_temp_work_dir is not None:
            dir = base_temp_work_dir
            warnings.warn("TempWorkDir 'base_temp_work_dir' argument is deprecated, use 'dir'", DeprecationWarning)
        if dir is not None:
            safemakedirs(dir)
        self.dirname = mkdtemp(dir=dir, prefix=prefix)
        self.cleanup = cleanup
        self.transfer = transfer
        self.chdir = chdir

    def __enter__(self):
        self.curdir = os.getcwd()
        if self.chdir:
            os.chdir(self.dirname)
        if self.transfer is not None:
            for file in self.transfer:
                if os.path.isfile(file):
                    shutil.copy(file, self.dirname)
                elif os.path.isdir(file):
                    shutil.copytree(file, os.path.join(self.dirname,
                                    file.split(os.sep)[-1]))
                else:
                    raise Exception('In otwrapy.TempWorkDir : the current '
                                    + 'path "{}" is not a file '.format(file)
                                    + 'nor a directory to transfer.')

        return self.dirname

    def __exit__(self, type, value, traceback):
        if self.chdir:
            os.chdir(self.curdir)
        if self.cleanup:
            shutil.rmtree(self.dirname)


def _exec_sample_serial(func, verbosity):
    """Return a function that evaluate the sample and provide a progress bar.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.
    verbosity : bool
        If True, the progress bar is displayed.

    Returns
    -------
    _exec_sample : Function or callable
        The function with the progress bar.
    """

    def _exec_sample(X):
        X = ot.Sample(X)
        if verbosity and X.getSize() > 1:
            Y = ot.Sample(0, func.getOutputDimension())
            for x in tqdm(X):
                Y.add(func(x))
        else:
            Y = func(X)
        return ot.Sample(Y)

    return _exec_sample


def _exec_sample_joblib(func, n_cpus, verbosity):
    """Return a function that executes a sample in parallel using joblib.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.
    n_cpus : int
        Number of CPUs on which to distribute the function calls.
    verbosity : bool
        If True, the progress bar is displayed.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """
    import joblib
    from joblib import Parallel, delayed
    import contextlib

    # solution found in: https://stackoverflow.com/questions/24983493/
    # tracking-progress-of-joblib-parallel-execution/58936697#58936697
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()

    def _exec_sample(X):
        if verbosity:
            with tqdm_joblib(tqdm(total=ot.Sample(X).getSize())):
                Y = Parallel(n_jobs=n_cpus, verbose=0)(delayed(func)(x) for x in X)
        else:
            Y = Parallel(n_jobs=n_cpus, verbose=0)(delayed(func)(x) for x in X)
        return ot.Sample(Y)

    return _exec_sample


def _exec_sample_multiprocessing(func, n_cpus, verbosity):
    """Return a function that executes a sample in parallel using multiprocessing.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.
    n_cpus : int
        Number of CPUs on which to distribute the function calls.
    verbosity : bool
        If True, the progress bar is displayed.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """
    def _exec_sample(X):
        from multiprocessing import Pool
        with Pool(processes=n_cpus) as p:
            if verbosity:
                rs = list(tqdm(p.imap(func, X), total=ot.Sample(X).getSize()))
            else:
                rs = list(p.imap(func, X))
        return ot.Sample(rs)
    return _exec_sample


def _exec_sample_pathos(func, n_cpus):
    """Return a function that executes a sample in parallel using pathos.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """
    def _exec_sample(X):
        from pathos.multiprocessing import ProcessingPool
        try:
            p = ProcessingPool(n_cpus)
            X = np.array(X)
            x = np.array_split(X, n_cpus)
            # array_split is not supposed to return a list of length n_cpus when len(X)<n_cpus
            n_active = min(len(X), n_cpus)
            pipe = []
            for i in range(n_active):
                pipe.append(p.apipe(func, x[i]))

            rs = []
            for i in range(n_active):
                rs.append(pipe[i].get())

            rs = [item for sublist in rs for item in sublist]

            return ot.Sample(rs)
        except ValueError:
            # Get there if the chuck size left some single evaluations left
            return func(X)
    return _exec_sample


def _exec_sample_pydefx(func, n_cpus):
    """Return a function that executes a sample in parallel using pydefx.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """
    def _exec_sample(X):
        import pydefx
        params = pydefx.Parameters("localhost", n_cpus)
        params.salome_parameters.work_directory = os.path.join(params.salome_parameters.work_directory, "otwrapy")
        params.createResultDirectory("/tmp")
        study = pydefx.PyStudy()
        local_work_dir = params.salome_parameters.result_directory
        Xs = ot.Sample(X)
        nb_inputs = Xs.getDimension()
        nb_outputs = func.getOutputDimension()
        args = ", ".join([f"p{i}" for i in range(nb_inputs)])
        outs = ", ".join([f"y{i}" for i in range(nb_outputs)])
        FUNCTION_DUMP_FILE_NAME = "function_dump.xml"
        callable_dump_file = os.path.join(local_work_dir, FUNCTION_DUMP_FILE_NAME)
        studyOT = ot.Study()
        studyOT.add("function", ot.Function(func))
        studyOT.setStorageManager(ot.XMLStorageManager(callable_dump_file))
        studyOT.save()
        params.salome_parameters.in_files.append(callable_dump_file)
        run_script = pydefx.PyScript()
        run_script.loadString(f"""
import openturns as ot
def _exec({args}):
    studyOT = ot.Study()
    studyOT.setStorageManager(ot.XMLStorageManager("{FUNCTION_DUMP_FILE_NAME}"))
    studyOT.load()
    function = ot.Function()
    studyOT.fillObject("function", function)
    X = [{args}]
    Y = function(X)
    {outs} = Y[0] if len(Y) == 1 else Y
    return {outs}
""")
        ydefx_sample = run_script.CreateEmptySample()
        dict_sample = {}
        for i in range(nb_inputs):
            dict_sample[f"p{i}"] = []
        for point in Xs:
            for i in range(nb_inputs):
                dict_sample[f"p{i}"].append(point[i])
        ydefx_sample.setInputValues(dict_sample)
        study.createNewJob(run_script, ydefx_sample, params)
        study.launch()
        study.wait()
        result = study.getResult()
        if result.hasErrors():
            print(result.getErrors())
            print(study.sample.getMessages())
            raise RuntimeError("Could not evaluate sample using pydefx backend")
        outS = ot.Sample(len(Xs), nb_outputs)
        for i in range(len(Xs)):
            for j in range(nb_outputs):
                outS[i, j] = study.sample.getOutput(f"y{j}")[i]
        return outS
    return _exec_sample


def _exec_sample_ipyparallel(func, n_cpus, ipp_client_kw):
    """Return a function that executes a sample in parallel using ipyparallel.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    ipp_client_kw : dict
        Client parameters

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """
    import ipyparallel as ipp

    client = ipp.Client(**ipp_client_kw)
    client.wait_for_engines(n_cpus, timeout=30)

    return lambda X: client[:].map_sync(func, X)


def _exec_sample_concurrent(func, n_cpus, executor_name, verbosity):
    """Return a function that executes a sample in parallel using concurrent thread or process.

    Parameters
    ----------
    func : Function or callable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.
    n_cpus : int
        Number of CPUs on which to distribute the function calls.
    executor_name : str
        Name of the executor to use : either "thread" or "process".
    verbosity : bool
        If True, the progress bar is displayed.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized function.
    """

    def _exec_sample(X):
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        pool_executor = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}

        if verbosity:
            pbar = tqdm(total=ot.Sample(X).getSize())
        with pool_executor[executor_name](max_workers=n_cpus) as executor:
            # resu = executor.map(func, X)
            # Y = ot.Sample([xx for xx in resu])
            resu = {executor.submit(func, x): x for x in X}
            for future in as_completed(resu):
                try:
                    _ = future.result()
                    if verbosity:
                        pbar.update(1)
                except Exception:
                    raise Exception('Error when evaluating %r. ' % resu[future])
            Y = [task.result() for task in resu]
        if verbosity:
            pbar.close()

        return Y
    return _exec_sample


def _exec_sample_dask_ssh(func, dask_args, verbosity):

    from dask.distributed import Client, progress, SSHCluster

    python_list = None
    if 'remote_python' in dask_args.keys():
        # start with python of the scheduler
        python_list = [dask_args['remote_python'][dask_args['scheduler']]]

    worker_list = []
    for worker, n_cpus in dask_args['workers'].items():
        worker_list.extend([worker] * n_cpus)

        if python_list is not None:
            # add python path of the workers
            python_list.extend([dask_args['remote_python'][worker]] * n_cpus)

    cluster = SSHCluster(
        [dask_args['scheduler']] + worker_list,
        connect_options={"known_hosts": None},
        worker_options={"nthreads": 1, "n_workers": 1},
        scheduler_options={"port": 0, "dashboard_address": ":8787"},
        remote_python=python_list)

    client = Client(cluster)

    def _exec_sample(X):
        map_eval = client.map(func, X)
        if verbosity:
            progress(map_eval)
        result = client.submit(list, map_eval)
        return ot.Sample(result.result())

    return _exec_sample, cluster, client


def _exec_sample_dask_slurm(func, n_cpus, slurmcluster_kw, verbosity):
    from dask.distributed import Client, progress
    from dask_jobqueue import SLURMCluster

    # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
    cluster = SLURMCluster(**slurmcluster_kw)

    # https://docs.dask.org/en/latest/futures.html#distributed.Client
    client = Client(cluster)

    def _exec_sample(X):
        map_eval = client.map(func, X)
        if verbosity:
            progress(map_eval)
        result = client.gather(map_eval)
        return ot.Sample(result)

    return _exec_sample, cluster, client


@FunctionDecorator(enableCache=True)
class Parallelizer(ot.OpenTURNSPythonFunction):

    """Parallelize a Wrapper using 'ipyparallel', 'joblib', 'pathos' or 'multiprocessing'.

    Parameters
    ----------

    wrapper : ot.Function or instance of ot.OpenTURNSPythonFunction
        openturns wrapper to be distributed

    backend : str, optional
        Whether to parallelize using 'ipyparallel', 'joblib', 'pathos',
        'multiprocessing', 'dask/ssh', 'dask/slurm', 'concurrent/thread', 'concurrent/process',
        'pydefx' or 'serial'.
        Default is multiprocessing.
        Also the backend will fallback to multiprocessing when the corresponding third-party
        cannot be imported.

    n_cpus : int, optional
        Number of CPUs on which the simulations will be distributed. Needed only
        if using 'joblib', pathos or 'multiprocessing' as backend.
        If n_cpus = 1, the behavior is the same as 'serial'.
        The default is -1, which means multiprocessing.cpu_count / 2
        Note that for remote/distributed backends this may not reflect the remote node capabilities.

    verbosity : bool, optional
        Whether to display a progress bar.
        Default is True.

    dask_args : dict, optional
        Dictionnary parameters when using Dask SSH Cluster. It must follow this form:
        {'scheduler': ip adress or host name,
        'workers': {'ip adress or host name': n_cpus},
        'remote_python': {'ip adress or host name': path_to_bin_python}}.
        The parallelization uses SSHCluster class of dask distributed with 1 thread per worker.
        When dask is chosen, the argument n_cpus is not used. The progress bar is enabled if
        verbosity is True.
        The dask dashboard is enabled at port 8787.

    slurmcluster_kw : dict, optional
        Parameters to instantiate the Dask SLURMCluster object.
        The argument n_cpus is used to set the default number of workers (n_workers).

    ipp_client_kw : dict, optional
        Parameters to instantiate the IPython Parallel Client, like "cluster_id", etc.

    Examples
    --------
    For example, in order to parallelize the beam wrapper :class:`examples.beam.Wrapper`
    you simply instantiate your wrapper and parallelize it as follows:

    >>> from otwrapy.examples.beam import Wrapper
    >>> import otwrapy as otw
    >>> model = otw.Parallelizer(Wrapper(), n_cpus=-1)

    `model` will distribute calls to Wrapper() using multiprocessing and
    as many CPUs as you have minus one for the scheduler.

    Because Parallelize is decorated with :class:`FunctionDecorator`,
    `model` is already an :class:`ot.Function`.
    """

    def __init__(self, wrapper, backend="multiprocessing", n_cpus=-1, verbosity=True,
                 dask_args=None, slurmcluster_kw={}, ipp_client_kw={}):

        # -1 cpus means all available cpus - 1 for the scheduler
        if n_cpus == -1:
            import multiprocessing
            n_cpus = max(multiprocessing.cpu_count() // 2 - 1, 1)

        self.n_cpus = n_cpus
        self.wrapper = wrapper
        self.verbosity = verbosity
        self.dask_args = dask_args
        # This configures how to run single point simulations on the model:
        self._exec = self.wrapper

        ot.OpenTURNSPythonFunction.__init__(self,
                                            self.wrapper.getInputDimension(),
                                            self.wrapper.getOutputDimension())

        self.setInputDescription(self.wrapper.getInputDescription())
        self.setOutputDescription(self.wrapper.getOutputDescription())

        if backend == "dask":
            backend = "dask/ssh"
            warnings.warn("'dask' backend is deprecated, use 'dask/ssh'", DeprecationWarning)
        if backend == "ipython":
            backend = "ipyparallel"
            warnings.warn("'ipython' backend is deprecated, use 'ipyparallel'", DeprecationWarning)

        assert backend in ["serial", "ipyparallel", "multiprocessing", "pathos", "pydefx",
                           "joblib", "dask/ssh", "dask/slurm",
                           "concurrent/thread", "concurrent/process"], f"Unknown backend: {backend}"

        # This configures how to run samples on the model :
        if backend == "serial" or self.n_cpus == 1:
            self._exec_sample = _exec_sample_serial(self.wrapper, self.verbosity)

        elif backend == "ipyparallel":
            self._exec_sample = _exec_sample_ipyparallel(self.wrapper, self.n_cpus, ipp_client_kw)

        elif backend == "joblib":
            self._exec_sample = _exec_sample_joblib(self.wrapper, self.n_cpus, self.verbosity)

        elif backend == "multiprocessing":
            self._exec_sample = _exec_sample_multiprocessing(
                self.wrapper, self.n_cpus, self.verbosity)

        elif backend == "pathos":
            self._exec_sample = _exec_sample_pathos(self.wrapper, self.n_cpus)

        elif backend == "concurrent/thread":
            self._exec_sample = _exec_sample_concurrent(
                self.wrapper, self.n_cpus, "thread", self.verbosity)

        elif backend == "concurrent/process":
            self._exec_sample = _exec_sample_concurrent(
                self.wrapper, self.n_cpus, "process", self.verbosity)

        elif backend == "pydefx":
            self._exec_sample = _exec_sample_pydefx(self.wrapper, self.n_cpus)

        elif backend == "dask/ssh":
            assert 'scheduler' in self.dask_args, 'dask_args must have "scheduler" as key'
            assert 'workers' in self.dask_args, 'dask_args must have "workers" as key'

            self._exec_sample, self.dask_cluster, self.dask_client = _exec_sample_dask_ssh(
                self.wrapper, self.dask_args, self.verbosity)

            def close_dask():
                from time import sleep
                self.dask_client.close()
                sleep(1)
                self.dask_cluster.close()

            self.close_dask = close_dask

        elif backend == "dask/slurm":

            slurmcluster_kw = dict(slurmcluster_kw)
            slurmcluster_kw.setdefault("n_workers", 1)
            slurmcluster_kw.setdefault("cores", n_cpus)
            slurmcluster_kw.setdefault("processes", n_cpus)
            slurmcluster_kw.setdefault("memory", "512 MB")
            self._exec_sample, self.dask_cluster, self.dask_client = _exec_sample_dask_slurm(
                self.wrapper, n_cpus, slurmcluster_kw, self.verbosity)
