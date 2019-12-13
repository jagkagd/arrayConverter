# np-xarr

Perform a numpy array transformation intuitively by giving simple patterns.

## Install

```shell script
$ pip install np-xarr
```

## Usage

```python
>> from npxarr import X
>> import numpy as np

>> a = X('[1, 2, 3, ...]', '[[1, 2], [2, 3], ...]')(np.r_[0, 1, 2, 3, 4, 5]) # sliding window

[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]

>> a = X('[[1, 2, ...], 
    [3, 4, ...], ...]', '[[1, 3, ...], [2, 4, ...], ...]') # transpose
>> a(np.array([[0, 1], [2, 3], [4, 5]])

[[0 2 4]
 [1 3 5]]
```

Multiple inputs or outputs are supported.

```python
>> a = X(['[1, 2, ...]', '[a, b, ...]'],  # multiple input in a list
         '[1, a, 2, b, ...]; [[a, 1], [b, 2], ...]') # or seperate by ;
>> a([np.r_[1, 2, 3, 4, 5], np.r_[10, 20, 30]]) # for incompatible input shapes, it will figure out the maximum valid output shape

(array([ 1, 10,  2, 20,  3, 30,  4], dtype=int32), 
 array([[10,  1], [20,  2], [30,  3]], dtype=int32))

>> a[1]([np.r_[1, 2], np.r_[10, 20, 30]) # or just get the transformation for second output

[[10  1], [20  2]]
```
Functions can be applied.
```python
>> a = X('[1, 2, 3, 4, ...]', '[times(2), neg(1), times(4), neg(3), ...]', 
         f={'neg': lambda x: -x, 'times': lambda x: 10*x})
```
notice here the output with sequence [2, 1, 4, 3, ...]
```python
>> a(np.r_[0, 1, 2, 3, 4, 5])

[10, 0, 30, -2, 50, -4]
```
and unpacking
```python
>> a = X('1; 2', '[*1, *2, *1]')([np.r_[1, 2],  np.r_[10, 20]])

[ 1  2 10 20  1  2]
```
You can provide output shape by hand
```python
>> a = X('[1, 2, ...]', '[[1, 1, ...], [2, 2, ...], ...]')
>> a(np.arange(6), outShapes=(-1, 3)) # or outShapes=[(-1, 3)], 

[[0 0 0]
 [1 1 1]
 [2 2 2]
 [3 3 3]
 [4 4 4]
 [5 5 5]]
```
And by providing parameter `extraShapes`...
```python
>> a = X('[1, 2, 3, ...]', '[[1, 2], [2, 3], ...]')
>> a(np.r_[0, 1, 2, 3], extraShapes=(1, 0)))

[[0 1]
 [1 2]
 [2 3]
 [3 0]]
```

## Notes:

* It is recommended to write patterns with at least two periods, e.g. [1, 2, ...] -> [[1, 2], ...] will be inferred as [1, 2, 3, ...] -> [[1, 2], [2, 3], ...] rather than [[1, 2], [3, 4], ...]

* Inefficient for large array

    The output array is built by code like `np.array([inArrays[indexConverter(index)] for index <= outShape])`

* Only support transformation with formula `$y_j = floor(a_ij*x_i) + b_j + floor(c_ij*mod(x_i, d_ij))$`

## Todo

- [ ] Test cover
- [ ] Improve exception system
- [ ] Support for illegal python variable name like `a.1`
- [ ] Try to deduce possible transformation using native numpy function from calculated equation
