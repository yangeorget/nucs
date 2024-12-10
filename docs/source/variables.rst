#########
Variables
#########

********************
Variables as indices
********************

In NuCS, variables are not entities in their own right, but simply domain indices.
As a consequence, variables do not have any attributes such as names, ...

The number of domains is an unsigned 16-bits integer.


****************************
Variable-domain relationship
****************************

Several variables can share/index a single domain.
The use of offsets makes it possible to implement the relationship :math:`x = y + c` without any constraint.


**************************
Decision variables/domains
**************************

During the resolution of a problem, decisions are made on a subset of the shared domains.
These domains are named decision domains or sometimes, by abuse of language, decision variables.




