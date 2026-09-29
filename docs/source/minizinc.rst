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

The search annotations of the model are followed exactly, with the MiniZinc restart annotations on the solve item:
:code:`restart_luby`, :code:`restart_geometric`, :code:`restart_linear`, :code:`restart_constant` and
:code:`restart_none`. A restart limit counts failures. Last-conflict has no annotation, so it has its own flag, and
:code:`--restart` sets or replaces the restart policy:

.. code-block:: bash

   minizinc --solver nucs --last-conflict model.mzn
   minizinc --solver nucs --restart luby,500 model.mzn

**Free search.** With :code:`-f`, NuCS keeps only which variables the search annotations name, and chooses the
order and the values itself: dom/wdeg on the annotated variables, then on the others, with last-conflict and Luby
restarts (scale 500). :code:`--no-last-conflict` and :code:`--restart` change these defaults.

.. code-block:: bash

   minizinc --solver nucs -f model.mzn
   minizinc --solver nucs -f --restart none model.mzn

On 41 MiniZinc challenge instances in 60 s, the free search found a better result than the models' own searches on
11 and a worse one on 5. The 5 are models whose search annotation carries knowledge that dom/wdeg does not have,
so :code:`-f` is not always the better choice.

A model that uses a builtin the adapter does not yet support exits with a clear
:code:`constraint '<name>' is not supported` message.


**********************
Use the MiniZinc IDE
**********************

Once you have run :code:`fzn-nucs --register`, NuCS appears automatically in the MiniZinc IDE
solver dropdown (the registration uses absolute paths in the standard user solvers directory,
so no extra IDE configuration is required).
