#########
Variables
#########

********************
Variables as indices
********************

In NuCS, variables are not entities in their own right, but simply domain indices.
As a consequence, variables do not have any attributes such as names, etc.

The domain indices are unsigned 16-bits integers.


****************************
Variable-domain relationship
****************************

There is no 1-1 relationship between variables and domains.
Instead, several variables can share/index a single domain.
Any modification to a shared domain benefits all variables sharing this domain.

Moreover, the use of offsets makes it possible to implement the relationship :math:`x = y + c` without any constraint.
Indeed, variables :math:`x` and :math:`y` share a single domain with offsets :math:`0` and :math:`c` (for example).

**************************
Decision variables/domains
**************************

During the resolution of a problem, decisions are made on a subset of the shared domains.
These domains are named decision domains or sometimes, by abuse of language, decision variables.




