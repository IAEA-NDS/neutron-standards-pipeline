# Validation of Python pipeline

## Scenario 0

For reference, the following sequence has been run with
the Fortran GMAP code and the Fortran DATP code:

1) Reduce the `input_std2017.crd` file with DATP
2) Perform an evaluation with GMAP (3 iterations, PPP correction)
3) Update prior in DAT.INP and perform another reduction with DATP
4) Perform another evaluation with GMAP (3 iterations, PPP correction)

This sequence has been automated with a Python script called `run_approach.py`.

TO ensure that the Python `datpy` code (modernized version of Fortran DATP) and
the Python `gmapy` code (modernized version of Fortran GMAP) work as expected,
a sequence of test scenarios has been investigated to understand precisely
to which extent the Python codes can reproduce the Fortran codes behavior,
and if differences occur, to precisely understand the reason for these differences.

For each scenario, there is an excel sheet with the corresponding number, e.g.
the results of scenario 1 are saved in `scenario_01.xlsx`.

## Scenario 1

The reduced GMA database resulting from step (3) is taken as starting point
for an evaluation performed with the Python legacy mode (`run_legacy_gmap`).
In this mode, all the bugs of the Fortran GMAP code are reproduced:

- PPP correction bug 
- SACS Jacobian computation bug
- SACS integration routine bug

The results of the Fortran GMAP (scenario 0) and the Python gmapy code
are numerically identical (relative difference below `1e-5`)

## Scenario 2

The reduction in step (3) is performed with the Python `datpy` code.
The equivalence of the Fortran `DATP` and the Python `datpy` has been
established independently. However, the `datpy` code can store the
reduced GMA database in a JSON file with extended output precision.
Performing then the evaluation with the Python `gmapy` code in
legacy mode (same as in scenario 1) using this JSON file leads to some
differences in the results. The largest discrepancy of 1% is for 10B(n,a0)
at 6.5 keV. Otherwise, the discrepancy for any other quantity stays below 0.3%.
Having established numerical equivalency between the Fortran GMAP and Python
gmapy code in scenario 1, the conclusion of scenario 2 is that the extended
output precision of `datpy` causes small changes in the ensuing evaluation.

## Scenario 3

We use all the ingredients of scenario 2 but run the Python legacy mode
*with all bugs switched off*. This slighlty increases the discrepancies
in the result between scenario 0. We can now say that most differences
are below 0.4% but except for 10B(n,a0) and 10B(n,a1) where some differences
are between 0.4% and 1%.

## Scenario 4

In scenario 4 we again use the same ingredients as in scenario 2
but use the modern Python mode (`run_gmap_simplified`). This mode
does not allow anymore to reproduce the bugs of the Fortran version.
The objective of this scenario is to establish that scenario 3
and scenario 4 yield the same result, in other words to know that
`run_gmap_simplified` is euivalent to the Fortran behavior without bugs.
This result, indeed, was obtained.

## Scenario 5

Again, same ingredients as in scenario 4, but we increase the number
of iterations to 10 in the evaluation with `run_gmap_simplified` to ensure
that we obtain the same result as with 3 iterations (as in scenario 4). 
The relative difference in the results is below 1e-4, which confirms
that 3 iterations yield sufficient convergence in practice.

## Scenario 6

Again same ingredients as in scenario 5, but we remove the so-called
dummy data in the evaluation (which is a reguarlization trick) and replace
this by a more conventional regularization (similar to how it is one in 
the Levenberg-Marquardt algorithm). This is to ensure that the dummy
data regularization does not have an impact on the convergence behavior.
The relative differences to scenario 5 are below 1e-9 except for 
6Li(n,a) at 0.98 MeV where it is 1.2% and at 6Li(n,n) at 0.98 MeV where it 
is 0.28%. This indicates that the regularization scheme does not seem to matter
except for some isolated cases.

## Scenario 7

Same ingredients as in scenario 6 (no dummy data, alternative regularization scheme),
but the evaluation is performed in Python TensorFlow mode. The result of scenario 6
and scenario 7 are identical (i.e. no relative difference whatsoever). This
demonstrates that the Python TensorFlow mode is fully compatible with the Fortran
version (but without bugs and without dummy data regularization).

## Scenario 8

Same ingredients as in scenario 7 but we use the Python TensorFlow mode to
perform chisquare minimization. We confirmed with an independent check
that the chisquare minimization was successful (by small element-wise perturbations).
The results differ quite significantly from scenario 7. In the energy region
up to 1 MeV, the discrepancy to scenario 7 for U8(n,f) can go as high as 30% and
does always amount to several percent.
In conclusion, iterative GLS does not minimize the chisquare and the resulting
cross sections for the two approaches can be quite different.

## Scenario 9

Same ingredients as in scenario 7 but we use the Python TensorFlow mode
to perform Maximum Likelihood Estimation (MLE).
TODO TODO

## Scenario 10

In this scenario we use the iterative reduction routine in Python relying
on TensorFlow and the iterative Generalized Least Squares routine.
We perform a single iteration for the reduction.
We expect similar results to scenario 6. We find that except for
6Li(n,n) at 0.98 MeV were the relative difference is 0.6%, the
relative difference is below 0.1% except for a couple of cases,
most notable U8(n,g) at low energies where the relative difference
can be around 0.2%.

## Scenario 11

Same ingredients as in scenario 10 but we perform 5 iterations for the reduction. 
Importantly, the results for some quantities after 5 reduction steps are
quite different compare to just one reduction step. For instance,
the relative difference between scenario 10 and 11 for U8(n,f) at 0.52 MeV is 17%.
About 263 result quantities (cross sections at various energies) differ by more
than 1%.

## Scenario 12

Same ingredients as 11 but we do now 10 reduction iterations.
The differences to scenario 11 are smaller than to 10 but still quite large.
The largest difference for Au(n,g) at 0.0035 MeV is about 12%.
A number of 80 result quantities differ by more than 1% from scenario 11. 

## Scenario 13

Same ingredients as scenario 10 (one reduction iteration)
but we use the chisquare minimization mode, both more reduction
and evaluation. Results can are quite different. Considering the
result table for details.

## Scenario 14

Same ingredients as scenario 10 (one reduction iteration)
but we use the Maximum Likelihood Estimation (MLE) mode,
both more reduction and evaluation. Results can be quite different.
Considering the result table for details.
