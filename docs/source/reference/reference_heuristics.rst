.. _heuristics:

**********
Heuristics
**********

NUCS provides heuristics for selecting a variable and for selecting a value (more generally, reducing its domain):

Heuristics for selecting a variable
###################################

NUCS provides the following functions for selecting a variable

.. py:module:: nucs.heuristics.first_not_instantiated_var_heuristic
.. py:function:: nucs.heuristics.first_not_instantiated_var_heuristic.first_not_instantiated_var_heuristic(decision_variables, shr_domains_stack, stacks_to, params)

   This heuristics chooses the first non-instantiated variable.

   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the variable
   :rtype: int


.. py:module:: nucs.heuristics.greatest_domain_var_heuristic
.. py:function:: nucs.heuristics.greatest_domain_var_heuristic.greatest_domain_var_heuristic(decision_variables, shr_domains_stack, stacks_top, params)

   This heuristics chooses the first non-instantiated variable with the greatest domain.

   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the variable
   :rtype: int


.. py:module:: nucs.heuristics.max_regret_var_heuristic
.. py:function:: nucs.heuristics.max_regret_var_heuristic.greatest_domain_var_heuristic(decision_variables, shr_domains_stack, stacks_top, params)

   This heuristics chooses the variable with the maximal regret (difference between best and second best values).

   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param params: a two-dimensional costs array, by variable then by value
   :type params: NDArray
   :return: the variables
   :rtype: int


.. py:module:: nucs.heuristics.smallest_domain_var_heuristic
.. py:function:: nucs.heuristics.smallest_domain_var_heuristic.smallest_domain_var_heuristic(decision_variables, shr_domains_stack, stacks_top, params)

   This heuristics chooses the first non-instantiated variable with the smallest domain.

   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the variables
   :rtype: int


Heuristics for reducing the chosen domain
#########################################

NUCS provides the following functions for reducing a domain.


.. py:module:: nucs.heuristics.max_value_dom_heuristic
.. py:function:: nucs.heuristics.max_value_dom_heuristic.max_value_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the last value of the domain.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param dom_idx: the index of the domain
   :type dom_idx: int
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the MIN event
   :rtype: int


.. py:module:: nucs.heuristics.mid_value_dom_heuristic
.. py:function:: nucs.heuristics.mid_value_dom_heuristic.min_value_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the middle value of the domain.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param variable: the variable
   :type variable: int
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: an event (MIN, MAX or MIN and MAX)
   :rtype: int


.. py:module:: nucs.heuristics.min_cost_dom_heuristic
.. py:function:: nucs.heuristics.min_cost_dom_heuristic.split_low_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the value of the domain that minimizes the cost.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param variable: the variable
   :type variable: int
   :param params: a two-dimensional costs array, by variable then by value
   :type params: NDArray
   :return: an event (MIN, MAX or MIN and MAX)
   :rtype: int


.. py:module:: nucs.heuristics.min_value_dom_heuristic
.. py:function:: nucs.heuristics.min_value_dom_heuristic.min_value_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the first value of the domain.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param variable: the variable
   :type variable: int
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the MAX event
   :rtype: int


.. py:module:: nucs.heuristics.split_low_dom_heuristic
.. py:function:: nucs.heuristics.split_low_dom_heuristic.split_low_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the first half of the domain.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param variable: the variable
   :type variable: int
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the MAX event
   :rtype: int


.. py:module:: nucs.heuristics.split_high_dom_heuristic
.. py:function:: nucs.heuristics.split_high_dom_heuristic.split_low_dom_heuristic(shr_domains_stack, dom_update_stacks, stacks_top, dom_idx, params)

   This heuristics chooses the second half of the domain.

   :param domains_stk: the stack of domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: the stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: the stack of domain updates
   :type dom_update_stk: NDArray
   :param stks_top: the index of the top of the stacks as a Numpy array
   :type stks_top: NDArray
   :param variable: the variable
   :type variable: int
   :param params: a two-dimensional parameters array, unused here
   :type params: NDArray
   :return: the MAX event
   :rtype: int

