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

Point MiniZinc at the bundled solver config and make sure :code:`fzn-nucs` is on your :code:`PATH`:

.. code-block:: bash

   # directory containing nucs.msc (inside the installed nucs package)
   export MZN_SOLVER_PATH="$(python -c 'import nucs.fzn, os; print(os.path.join(os.path.dirname(nucs.fzn.__file__), "share"))')"
   # persist the Numba JIT cache so runs after the first are fast
   export NUMBA_CACHE_DIR=.numba/cache

Check that NuCS is registered:

.. code-block:: bash

   minizinc --solvers

NuCS should appear in the list as :code:`NuCS 11.2.0 (org.nucs.nucs, cp, int)`.


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

In the MiniZinc IDE, add the same directory under
*Preferences → Additional solver search paths*, and launch the IDE from a shell
where :code:`fzn-nucs` is on your :code:`PATH`. NuCS then appears in the solver dropdown.
