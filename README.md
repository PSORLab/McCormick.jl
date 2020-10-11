# McCormick.jl
A Forward McCormick Operator Library

| **Linux/OS/Windows**                                                             |         **Coverage**                                                                    
|:--------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| [![Build Status](https://travis-ci.org/PSORLab/McCormick.jl.svg?branch=master)](https://travis-ci.org/PSORLab/McCormick.jl)  | [![Coverage Status](https://coveralls.io/repos/github/PSORLab/McCormick.jl/badge.svg?branch=master)](https://coveralls.io/github/PSORLab/McCormick.jl?branch=master) |

McCormick.jl is a component package in the EAGO ecosystem and is reexported by [EAGO.jl](https://github.com/PSORLab/EAGO.jl). It contains a library of forward McCormick operators (both nonsmooth and differentiable). Documentation for this is included in the [EAGO.jl](https://github.com/PSORLab/EAGO.jl) package and additional usage examples are included [EAGO-notebooks](https://github.com/PSORLab/EAGO-notebooks) in the form of Jupyter notebooks.

## McCormick operator variants

Each McCormick object is associated with a
parameter `T <: RelaxTag` which is either `NS` for nonsmooth relaxations ([Mitsos2009](https://epubs.siam.org/doi/abs/10.1137/080717341), [Scott2011](https://link.springer.com/article/10.1007/s10898-011-9664-7)), `MV` for multivariate relaxations ([Tsoukalas2014](https://link.springer.com/article/10.1007/s10898-014-0176-0), [Najman2017](https://link.springer.com/article/10.1007/s10898-016-0470-0)),
and `Diff` for differentiable relaxations ([Khan2016](https://link.springer.com/article/10.1007/s10898-016-0440-6), [Khan2018](https://link.springer.com/article/10.1007/s10898-017-0601-2), [Khan2019](https://www.tandfonline.com/doi/abs/10.1080/02331934.2018.1534108)). Conversion between `MV`, `NS`, and `Diff` relax tags are not currently supported. Convex and concave envelopes are used to compute relaxations of univariate functions.

## **Supported Operators**

In addition, to supporting the implicit relaxation routines of ([Stuber 2015](https://www.tandfonline.com/doi/abs/10.1080/10556788.2014.924514?journalCode=goms20)). This package
supports the computation of convex/concave relaxations (and asssociated subgradients) for
expressions containing the following operations:

**Common algebraic expressions**: `inv`, `log`, `log2`, `log10`, `exp`, `exp2`, `exp10`,
`sqrt`, `+`, `-`, `^`, `min`, `max`, `/`, `*`, `abs`, `step`, `sign`, `deg2rad`, `rad2deg`, `abs2`

**Trignometric Functions**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sec`, `csc`, `cot`, `asec`, `acsc`, `acot`, `sind`, `cosd`, `tand`, `asind`, `acosd`, `atand`, `secd`, `cscd`, `cotd`, `asecd`, `acscd`, `acotd`

**Hyperbolic Functions**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `sech`, `csch`, `coth`, `acsch`, `acoth`

**Special Functions**: `erf`, `erfc`, `erfcinv`, `erfc`

**Activation Functions**: `relu`, `leaky_relu`, `param_relu`, `sigmoid`, `bisigmoid`,
                          `softsign`, `softplus`, `maxtanh`, `pentanh`, `gelu`,
                          `elu`, `selu`, `swish1`

**Bound Specification Functions**: `positive`, `negative`, `lower_bnd`, `upper_bnd`, `bnd`

**Other Functions**: `one`, `zero`, `intersect`, `real`, `dist`, `eps`

Differentiable relaxations (`Diff <: RelaxTag`) are supported for the functions given in [Khan2016](https://link.springer.com/article/10.1007/s10898-016-0440-6), [Khan2018](https://link.springer.com/article/10.1007/s10898-017-0601-2), [Khan2019](https://www.tandfonline.com/doi/abs/10.1080/02331934.2018.1534108). However, differentiable relaxations for other nonsmooth terms listed above have yet to be developed and as such have been omitted.

## **Bounding a function via McCormick operators**
In order to bound a function using a McCormick relaxation. You first construct
structure that bounds the input variables then you construct pass these variables
two a function.

In the example below, convex/concave relaxations of the function `f(x) = sin(2x) + exp(x) - x`
are calculated at `x = 1` on the interval `[-2,3]`.

```julia
using McCormick

# create MC object for x = 2.0 on [1.0,3.0] for relaxing
# a function f(x) on the interval Intv

f(x) = x*(x-5.0)*sin(x)

x = 2.0                          # value of independent variable x
Intv = Interval(1.0,4.0)         # define interval to relax over
                                 # Note that McCormick.jl reexports IntervalArithmetic.jl
                                 # and StaticArrays. So no using statement for these is
                                 # necessary.
# create McCormick object
xMC = MC{1,NS}(x,Intv,1)

fMC = f(xMC)             # relax the function

cv = fMC.cv              # convex relaxation
cc = fMC.cc              # concave relaxation
cvgrad = fMC.cv_grad     # subgradient/gradient of convex relaxation
ccgrad = fMC.cc_grad     # subgradient/gradient of concave relaxation
Iv = fMC.Intv           # retrieve interval bounds of f(x) on Intv
```

The plotting the results we can easily generate visual the convex and concave
relaxations, interval bounds, and affine bounds constructed using the subgradient
at the middle of X.

![Figure_1](Figure_1.png)

This can readily be extended to multivariate functions as shown below

```julia

f(x) = max(x[1],x[2])

x = [2.0 1.0]                                    # values of independent variable x
Intv = [Interval(-4.0,5.0), Interval(-5.0,3.0)]  # define intervals to relax over

# create McCormick object
xMC = [MC{2,Diff}(x[i], Intv[i], i) for i=1:2)]

fMC = f(xMC)            # relax the function

cv = fMC.cv              # convex relaxation
cc = fMC.cc              # concave relaxation
cvgrad = fMC.cv_grad     # subgradient/gradient of convex relaxation
ccgrad = fMC.cc_grad     # subgradient/gradient of concave relaxation
Iv = fMC.Intv            # retrieve interval bounds of f(x) on Intv
```

![Figure_3](Figure_3.png)

### References
- **Khan KA, Watson HAJ, Barton PI (2017).** Differentiable McCormick relaxations. *Journal of Global Optimization*, 67(4):687-729.
- **Khan KA, Wilhelm ME, Stuber MD, Cao H, Watson HAJ, Barton PI (2018).** Corrections to: Differentiable McCormick relaxations. *Journal of Global Optimization*, 70(3):705-706.
- **Khan KA (2019).** Whitney differentiability of optimal-value functions for bound-constrained convex programming problems. *Optimization* 68(2-3): 691-711
- **Mitsos A, Chachuat B, and Barton PI. (2009).** McCormick-based relaxations of algorithms. *SIAM Journal on Optimization*, 20(2):573–601.
- **Najman J, Bongratz D, Tsoukalas A, and Mitsos A (2017).** Erratum to: Multivariate McCormick relaxations. *Journal of Global Optimization*, 68:219-225.
- **Scott JK,  Stuber MD, and Barton PI. (2011).** Generalized McCormick relaxations. *Journal of Global Optimization*, 51(4):569–606.
- **Stuber MD, Scott JK, Barton PI (2015).** Convex and concave relaxations of implicit functions. *Optim. Methods Softw.* 30(3), 424–460
- **Tsoukalas A and Mitsos A (2014).** Multivariate McCormick Relaxations. *Journal of Global Optimization*, 59:633–662.
- **Wechsung A, Scott JK, Watson HAJ, and Barton PI. (2015).** Reverse propagation of McCormick relaxations. *Journal of Global Optimization* 63(1):1-36.
