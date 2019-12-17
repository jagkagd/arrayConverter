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
or a simpler form:
```python
>> a = X('[[a0, a1], [a1, a2], ...]')
>> a = X('[[a00, a10, ...], [a01, a11, ...], ...]')
```
where `a` denotes the first input array, the number behind it denotes the index of the item.
For example, `a01` means the item in the first input array with index `(0, 1)`.


Multiple inputs or outputs are supported.

```python
>> a = X(['[1, 2, ...]', '[a, b, ...]'],  # multiple input in a list
         '[1, a, 2, b, ...]; [[a, 1], [b, 2], ...]') # or seperate by ;
# or >> a = X('[a0, b0, a1, b1, ...]; [[b1, a0], [b1, a1], ...]')
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
# a = X('[times(a1), neg(a0), times(a3), neg(a2), ...]')
```
notice here the output with sequence [2, 1, 4, 3, ...]
```python
>> a(np.r_[0, 1, 2, 3, 4, 5])

[10, 0, 30, -2, 50, -4]
```
and unpacking
```python
>> a = X('1; 2', '[*1, *2, *1]')([np.r_[1, 2],  np.r_[10, 20]])
# a = X('[*a, *b, *a]')

[ 1  2 10 20  1  2]
```
You can provide output shape by hand
```python
>> a = X('[1, 2, ...]', '[[1, 1, ...], [2, 2, ...], ...]')
# a = X('[[a0, a0, ...], [a1, a1, ...], ...]')
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
# a = X('[[a0, a1], [a1, a2], ...]')
>> a(np.r_[0, 1, 2, 3], extraShapes=(1, 0)))

[[0 1]
 [1 2]
 [2 3]
 [3 0]]
```

## How np-xarr does

When the pattern is given, e.g.,

```python
>> a = X('[a, b, c, ...]', '[[a, b], [b, c], ...]')
```
`X` will deduce the transformation equation between the input and output, and can be seen by
```python
>> a

y0 = |_x0_| + |_x1_|
```
where `|_x0_|` means `floor(x0)`.
The equation `y0 = |_x0_| + |_x1_|` build the relation between the output index `(x0, x1)` and the input index `(y0,)` as follows:

output index (x0, x1) | item | equation (x0, x1) -> (y0, ) | index (y0, ) | input item |
--------------------- | ---- | -------- | ------------------ | ---- |
(0, 0) | a | 0 + 0 = 0 | (0, ) | a
(0, 1) | b | 0 + 1 = 1 | (1, ) | b
(1, 0) | b | 1 + 0 = 1 | (1, ) | b
(1, 1) | c | 1 + 1 = 2 | (2, ) | c

Another example:
```python
>> a = X('[a, b, ...]', '[a, a, b, b, ...]')
>> a

y0 = |_0.50*x0_|
```

output index (x0, ) | item | equation (x0, ) -> (y0, ) | input index (y0, ) | item |
------------------- | ---- | -------- | ------------------ | ---- |
(0, ) | a | floor(0.5*0) = 0 | (0, ) | a
(1, ) | a | floor(0.5*1) = 0 | (0, ) | a
(2, ) | b | floor(0.5*2) = 1 | (1, ) | b
(3, ) | b | floor(0.5*3) = 2 | (1, ) | b

## Notes:

* It is recommended to write patterns with at least two periods, e.g. [1, 2, ...] -> [[1, 2], ...] will be inferred as [1, 2, 3, ...] -> [[1, 2], [2, 3], ...] rather than [[1, 2], [3, 4], ...]

* Inefficient for large array

    The output array is built by code like `np.array([inArrays[indexConverter(index)] for index <= outShape])`

* Only support transformation with formula `$y_j = floor(a_ij*x_i) + b_j + floor(c_ij*mod(x_i, d_ij))$`

## Todo

- [ ] Improve exception system
- [ ] Try to deduce possible transformation using native numpy function from calculated equation
