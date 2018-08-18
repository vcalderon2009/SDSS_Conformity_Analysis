.. _methods-mcf:

===========================
Marked Correlation Function
===========================

In the analysis of galactic conformity, we make use of the
*marked correlation function* (MCF, see
`Skibba et al (2006) <http://adsabs.harvard.edu/abs/2006MNRAS.369...68S>`_
for more information).

The MCF has the format of

.. math::
    \mathcal{M}(r_{p}) = \frac{1 + W(r_{p})}{1 + \xi(r_{p})} \equiv \frac{WW}{DD}


where :math:`\xi(r_{p})` is the usual two-point correlation function with pairs
summed up in bins of projected separation, :math:`r_{p}`, and :math:`W (r_{p})`
is the same except that galaxy pairs are weighted by the product of their marks.
The estimator used in the equation can also be written as :math:`WW/DD`,
where :math:`DD` is the raw number of galaxy pairs separated by :math:`r_{p}`
and :math:`WW` is the weighted number of pairs.
In the conformity analysis using MCF, we normalize by the mean of the galaxy
property and then compute the MCF results.
