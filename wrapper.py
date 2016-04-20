#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General purpose OpenTURNS python wrapper
"""

import openturns as ot
ot.ResourceMap.SetAsUnsignedInteger('cache-max-size', int(1e6))

import gzip
import os
import numpy as np
from xml.dom import minidom
import time
import otwrapy as otw

__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2015, Phimeca Engineering"
__version__ = "0.1.1"
__email__ = "aguirre@phimeca.fr"


class Wrapper(ot.OpenTURNSPythonFunction):
    """
    This Wrapper is intended to be lightweight so that it can be easily
    distributed across several nodes in a cluster.
    """

    # Possible configurations
    places = ['phimeca', 'poincare', 'tgcc']

    def __init__(self, where='phimeca', sleep=0.0):
        """
        Parameters
        ----------
        where : string or dict (Optional)
            Setup configuration according to where you run it.
        """

        assert where in Wrapper.places, "Only valid places are {}".format(Wrapper.places)
        self.base_dir = os.path.join(otw.base_dir, 'beam')

        if where == 'phimeca':
            self.temp_work_dir = '/tmp'
        if where == 'poincare':
            self.temp_work_dir = '/tmp'
        if where == 'tgcc':
            self.temp_work_dir = '/ccc/scratch/cont003/xxx/aguirref/Formation-PRACE/'

        self.input_template = os.path.join(self.base_dir, 'beam_input_template.xml')
        self.executable = os.path.join(self.base_dir, 'beam -x beam.xml')
        self.sleep = sleep

        # Number of input/output values:
        super(Wrapper, self).__init__(4, 1)
        self.setInputDescription(['Load', 'Young modulus', 'Length', 'Inertia'])
        self.setOutputDescription(['deviation'])

    def _exec(self, X):
        """Run the model in the shell.

        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated
        """

        # Create intentional delay
        time.sleep(self.sleep)

        # File management. Move to temp work dir. Cleanup at the end
        with otw.TempWorkDir(self.temp_work_dir, 'ot-beam-example-', True):
        
            # Create input file
            self._create_input_file(X)

            # Execute code
            runtime = self._call(X)

            # Retrieve output (see also ot.coupling_tools.get_value)
            Y = self._parse_output()

        return Y

    def _create_input_file(self, X):
        """Create the input file required by the code.

        Replace the values of the vector X to their corresponding tokens on the
        self.input_template and create the input file `beam.xml` on the current
        working directory.

        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated
        """
        ot.coupling_tools.replace(
            self.input_template,
            'beam.xml',
            ['@F','@E','@L','@I'],
            X)

    def _call(self, X):
        """Execute code on the shell
        
        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated

        Returns
        -------
        runtime : float
            Total runtime (wall time and not cpu time)
        """

        time_start = time.time()
        ot.coupling_tools.execute(self.executable)
        time_stop = time.time()

        return time_stop - time_start


    def _parse_output(self):
        """Parse the output given by the code

        Returns
        -------
        Y : list
            Output vector of the model. Univariate in this case.
        """

        # Retrieve output (see also coupling_tools.get_value)
        xmldoc = minidom.parse('_beam_outputs_.xml')
        itemlist = xmldoc.getElementsByTagName('outputs')
        deviation = float(itemlist[0].attributes['deviation'].value)

        # Make a list out of the output(s)
        Y = [deviation]

        return Y

####################################################################
# ------------------------ Parallel Wrapper ------------------------
####################################################################

@otw.NumericalMathFunctionDecorator(enableCache=True)
class ParallelWrapper(ot.OpenTURNSPythonFunction):
    """
    Class that distributes calls to the class Wrapper across a cluster using
    either 'ipython', 'joblib' or 'multiprocessing'.
    """
    def __init__(self, where='phimeca', backend='joblib',
        n_cpus=10, sleep=0.0):
        """
        Parameters
        ----------

        where : string (Optional)
            Setup configuration according to where you run it.

        backend : string (Optional)
            Whether to parallelize using 'ipython', 'joblib' or 'multiprocessing'.

        n_cpus : int (Optional)
            Number of CPUs on which the simulations will be distributed. Needed Only
            if using 'joblib' or 'multiprocessing' as backend.

        sleep : float (Optional)
            Intentional delay (in seconds) to demonstrate the effect of
            parallelizing.
        """

        self.n_cpus = n_cpus
        self.wrapper = Wrapper(where=where, sleep=sleep)
        # This configures how to run single point simulations on the model :
        self._exec = self.wrapper

        ot.OpenTURNSPythonFunction.__init__(self,
                self.wrapper.getInputDimension(),
                self.wrapper.getOutputDimension())

        self.setInputDescription(self.wrapper.getInputDescription())
        self.setOutputDescription(self.wrapper.getOutputDescription())

        # This configures how to run samples on the model :
        if self.n_cpus == 1:
            self._exec_sample = self.wrapper
        elif backend == 'ipython':
            self._exec_sample = otw._exec_sample_ipyparallel(self.wrapper,
                self.getInputDimension(), self.getOutputDimension())
        elif backend == 'joblib':
            self._exec_sample = otw._exec_sample_joblib(self.wrapper, self.n_cpus)
        elif backend == 'multiprocessing':
            self._exec_sample = otw._exec_sample_multiprocessing(self.wrapper, self.n_cpus)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description="Python wrapper example used for the PRACE training on HPC and uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-where', default='poincare', type=str, choices=['phimeca', 'poincare', 'tgcc'],
        help='Place where simulations will run.')
    
    parser.add_argument('-seed', default=int(0), type=int,
        help='Seed number for the random number generator')
        
    parser.add_argument('-MonteCarlo', nargs=1,
        help="Launch a MonteCarlo simulation of given size")
    
    parser.add_argument('-X', nargs='*',
        help='List of floats [X1, X2.. Xp] or PATH to a pickled DOE')
    
    parser.add_argument('-n_cpus', default=-1, type=int,
        help="(Optional) number of cpus to use.")

    parser.add_argument('-backend', default='joblib', type=str,
        choices=['joblib', 'multiprocessing'],
        help="Whether to parallelize using 'joblib' or 'multiprocessing'.")

    parser.add_argument('-run', default=False, type=bool, nargs='?', const='True',
        help='If True, run the model', choices=[True, False])

    parser.add_argument('-dump', default=False, type=bool, nargs='?', const='True',
        help='If True, dump the output for later posttreatment', choices=[True, False])

    args = parser.parse_args()

    model = ParallelWrapper(where=args.where, backend=args.backend, 
        n_cpus=args.n_cpus)
    print "The wrapper has been instantiated as 'model'."

    if args.MonteCarlo is not None:
        from probability_model import X_distribution
        ot.RandomGenerator.SetSeed(args.seed)
        N = int(args.MonteCarlo[0])
        X = X_distribution.getSample(N)
        print "Generated a MonteCarlo DOE of size {}".format(N)

    elif args.X is not None:
        if isinstance(args.X[0], str) and os.path.isfile(args.X[0]):
            X = otw.load_array(args.X[0])
            print "Loaded a DOE of size {} from file: '{}'".format(X.getSize(),
                args.X[0])
        else:
            X = ot.NumericalPoint([float(x) for x in args.X])


    if args.run:
        Y = model(X)
        # Dump the results if asked
        if args.dump:
            otw.dump_array(Y, 'OutputSample.pkl')
            print "The output has been saved to 'OutputSample.pkl'"
        else:
            print "Finished evaluationg the model. Take a look at 'Y' variable."
    elif (args.MonteCarlo is not None) or (args.X is not None):
        print "The desired input is ready to be run using --> 'Y = model(X)'"


