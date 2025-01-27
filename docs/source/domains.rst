#######
Domains
#######


*****************
Supported domains
*****************

NuCS supports integer and boolean domains.


Integer domains
###############

Domains bounds are 32-bits integers.


Boolean domains
###############

Boolean domains are integer domains of the form :math:`[0, 1]`.


******************
Domain persistence
******************

Shared domains, domain indices and offsets are stored using :code:`numpy.ndarray`.
Variable domains are not persisted but are computed on the fly using offsets and shared domains.
