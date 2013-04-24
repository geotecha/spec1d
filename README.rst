======
spec1d
======
Multilayer vertical and radial soil consolidation using the spectral method (one dimensional equal strain eta method)
---------------------------------------------------------------------------------------------------------------------
Features:

- Multiple soil layers
- Soil properties vary linearly within a layer (permeability, compressibility, lumped radial drainage parameter)
- Surcharge load varies piecewise linear with time.  Surcharge varies linearly within a layer, ratio between layers is constant with time.accept2dyear
- Pervious top, impervious bottom OR Pervious top, pervious bottom
- Top and bottom boundary pore pressure varies piecewise linear with time
- Output is pore pressure at a particular depth, average pore pressure between two points, settlement between two points (calculated using mv)

Ideas developed from [#]_, [#]_, [#]_, [#]_.

.. [#] Walker, Rohan. 2006. 'Analytical Solutions for Modeling Soft Soil Consolidation by Vertical Drains'. PhD Thesis, Wollongong, NSW, Australia: University of Wollongong.
.. [#] Walker, R., and B. Indraratna. 2009. 'Consolidation Analysis of a Stratified Soil with Vertical and Horizontal Drainage Using the Spectral Method'. Geotechnique 59 (5): 439-449.
.. [#] Walker, Rohan, Buddhima Indraratna, and Nagaratnam Sivakugan. 2009. 'Vertical and Radial Consolidation Analysis of Multilayered Soil Using the Spectral Method'. Journal of Geotechnical and Geoenvironmental Engineering 135 (5): 657-663
.. [#] Walker, Rohan T. 2011. 'Vertical Drain Consolidation Analysis in One, Two and Three Dimensions'. Computers and Geotechnics 38 (8): 1069-1077.
"""