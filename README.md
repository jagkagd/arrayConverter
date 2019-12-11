# arrayConverter

Perform a numpy array transformation by giving an example.

Examples:

```
>> from arrayConverter import X
>> import numpy as np
>> a = X('[1, 2, ...]', '[[1], [2], ...]') # [1, 2, ...] -> [[1], [2], ...]
>> a # print the equation from output index (x0, x1, ...) to input index (y0, y1, ...)
y0 = |_x0_| + |_x1_| # |_x_| means floor(x)
>> a(np.arange(6)) # apply the convertion to a certain numpy array
[[0], [1], [2], [3], [4], [5]]
```

Multiple inputs or outputs are supported.

```
>> a = X(['[[1], [2], ...]', '[a, b, ...]'],  # multiple input in a list
         '[[[1], [a]], [[2], [b]], ...]; [a, 1, b, 2, ...]') # or seperate by ;
>> a
out0: s = mod(|_x1_| + |_x2_|, 2) # s decides which input array
[
  in0: y0 = |_x0_| + |_x1_| + |_x2_|, y1 = |_x1_| + |_x2_| # (x0, x1, x2) -> (y0, y1)
  in1: y0 = |_x0_| + |_x1_| + |_x2_| -1
]
out1: s = mod(-|_mod(x0, 2)_| + 1, 2)
[
  in0: y0 = |_0.50*x0_|, y1 = 0
  in1: y0 = |_0.50*x0_|
]
>> a([np.array([[i] for i in range(6)]), np.arange(10)]) # notice incompatible input shape
(array([[[ 0], [ 0]],
       [[ 1], [-1]],
       [[ 2], [-2]],
       [[ 3], [-3]],
       [[ 4], [-4]],
       [[ 5], [-5]]]),
 array([ 0,  0, -1,  1, -2,  2, -3,  3, -4,  4, -5,  5, -6]))
# it will figure out the maximum valid output shape
```

Functions can be applied.
```
>> a = X('[1, 2, 3, 4, ...]', '[[times(2)], [neg(1)], [times(4)], [neg(3)], ...]', 
         f={'neg': lambda x: -x, 'times': lambda x: 10*x})
```
notice here the output with sequence [2, 1, 4, 3, ...]
```
>> a
y0 = |_x0_| + |_x1_| - |_2*mod(x0, 2)_| + 1
funcs: 
[
  times: mod(|_x1_| - |_mod(x0, 2)_| + 1, 2) = 1
  neg: mod(|_x1_| + |_mod(x0, 2)_|, 2) = 1
]
>> a(np.arange(6))
[[10], [ 0], [30], [-2], [50], [-4]]
```
You can provide output shape by hand
```
>> a = X('[1, 2, ...]', '[[1, 1, ...], [2, 2, ...], ...]')
>> a
y0 = |_x0_|
>> a(np.arange(6), outShapes=(-1, 3)) # or outShapes=[(-1, 3)], 
[[0 0 0]
 [1 1 1]
 [2 2 2]
 [3 3 3]
 [4 4 4]
 [5 5 5]]
```
And by providing parameter `extraShapes`...
```
>> a = X('[1, 2, 3, ...]', '[[1, 2], [2, 3], ...]')
>> a(np.arange(4), extraShapes=(1, 0)))
[[0 1]
 [1 2]
 [2 3]
 [3 0]]
```
