########################
Using NuCS from MiniZinc
########################

NuCS ships a `FlatZinc <https://docs.minizinc.dev/en/latest/fzn-spec.html>`_ adapter,
so you can model in `MiniZinc <https://www.minizinc.org/>`_ and solve with NuCS
via :code:`minizinc --solver nucs`.

Installing NuCS provides the :code:`fzn-nucs` executable and a MiniZinc solver configuration.


*********************
Register the solver
*********************

Run the one-off registration command:

.. code-block:: bash

   fzn-nucs --register

This writes a resolved ``nucs.msc`` into MiniZinc's user solvers directory
(:code:`~/.minizinc/solvers` on Linux/macOS, :code:`%APPDATA%\\MiniZinc\\solvers` on Windows),
with the version taken from the installed package and absolute :code:`executable` and :code:`mznlib`
paths, so no environment variable is needed.

Check that NuCS is registered:

.. code-block:: bash

   minizinc --solvers

NuCS should appear in the list as :code:`NuCS <version> (org.nucs.nucs, cp, int)`.

.. note::
   Re-run :code:`fzn-nucs --register` after upgrading NuCS or recreating the virtual environment,
   so the recorded version and paths stay correct.

For a temporary, non-persistent alternative, point MiniZinc at the bundled config instead:

.. code-block:: bash

   export MZN_SOLVER_PATH="$(python -c 'import nucs.fzn, os; print(os.path.join(os.path.dirname(nucs.fzn.__file__), "share"))')"


***************
Solve a model
***************

.. code-block:: bash

   minizinc --solver nucs model.mzn      # first solution
   minizinc --solver nucs -a model.mzn   # all solutions
   minizinc --solver nucs -n 5 model.mzn # first 5 solutions
   minizinc --solver nucs -s model.mzn   # with statistics on stderr

.. note::
   The first invocation is a few seconds slower while Numba compiles the propagators.
   With :code:`NUMBA_CACHE_DIR` set, later runs reuse the cache.

A model that uses a builtin the adapter does not yet support exits with a clear
:code:`constraint '<name>' is not supported` message.


**********************
Use the MiniZinc IDE
**********************

Once you have run :code:`fzn-nucs --register`, NuCS appears automatically in the MiniZinc IDE
solver dropdown (the registration uses absolute paths in the standard user solvers directory,
so no extra IDE configuration is required).
