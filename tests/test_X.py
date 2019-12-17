import numpy as np
from numpy.testing import assert_equal

from src.npxarr import X


def test_ZeroIndexConverter_1in():
    assert_equal(X('1', '[1, 1]')(np.r_[1, 2]), np.array([[1, 2], [1, 2]]))


def test_ZeroIndexConverter_1in_label():
    assert_equal(X('[a, a]')(np.r_[1, 2]), np.array([[1, 2], [1, 2]]))


def test_ZeroIndexConverter_2in():
    assert_equal(X('1;2', '[1, 2]')([np.r_[1, 2], np.r_[3, 4]]), np.array([[1, 2], [3, 4]]))


def test_ZeroIndexConverter_2in_label():
    assert_equal(X('[a, b]')([np.r_[1, 2], np.r_[3, 4]]), np.array([[1, 2], [3, 4]]))


def test_starred_2in():
    assert_equal(X('1;2', '[*1, *2]')([np.r_[1, 2], np.r_[3, 4]]), np.array([1, 2, 3, 4]))


def test_starred_2in_label():
    assert_equal(X('[*a, *b]')([np.r_[1, 2], np.r_[3, 4]]), np.array([1, 2, 3, 4]))


def test_starred():
    assert_equal(X('[1, ...]', '[*1, ...]')(np.array([[1, 2], [2, 3], [3, 4]])), np.array([1, 2, 2, 3, 3, 4]))


def test_starred_label():
    assert_equal(X('[*a0, ...]')(np.array([[1, 2], [2, 3], [3, 4]])), np.array([1, 2, 2, 3, 3, 4]))


def test_2in_outShape():
    assert_equal(X('[1, 2, ...]; [a, b, ...]', '[1, a, 2, b, ...]')([np.r_[1, 2, 3, 4], np.r_[-1, -2, -3]]),
                 np.r_[1, -1, 2, -2, 3, -3, 4])


def test_2in_outShape_label():
    assert_equal(X('[a0, b0, a1, b1, ...]')([np.r_[1, 2, 3, 4], np.r_[-1, -2, -3]]),
                 np.r_[1, -1, 2, -2, 3, -3, 4])


def test_2in_list():
    assert_equal(X(['[1, 2, ...]', '[a, b, ...]'], '[1, a, 2, b, ...]')([np.r_[1, 2, 3, 4], np.r_[-1, -2, -3]]),
                 np.r_[1, -1, 2, -2, 3, -3, 4])


def test_2in_outShape2():
    assert_equal(X('[1, 2, ...]; [a, b, ...]', '[1, a, 2, b, ...]')([np.r_[1, 2, 3, 4], np.r_[-1, -2]]),
                 np.r_[1, -1, 2, -2, 3])


def test_2in_subarray():
    assert_equal(X('[1, 2, ...]; [a, b, ...]', '[1, a, 2, b, ...]')(
        [np.array([[1, 2], [3, 4]]), np.array([[-1, -2], [-3, -4], [-5, -6]])]),
        np.array([[1, 2], [-1, -2], [3, 4], [-3, -4]]))


def test_2in_1use():
    assert_equal(X('[1, 2, ...]; [a, b, ...]', '[[1, 1], [2, 2], ...]')(
        [np.array([1, 2, 3, 4]), np.array([[-1, -2], [-3, -4], [-5, -6]])]),
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))


# def test_2in_1use_label():
#     with
#     assert_equal(X('[[a0, a0], [a1, a1], ...]')(
#         [np.array([1, 2, 3, 4]), np.array([[-1, -2], [-3, -4], [-5, -6]])]),
#         np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

def test_2out():
    a = X('[1, 2, ...]; [a, b, ...]', '[1, a, 2, b, ...]; [[a, 1], [b, 2], ...]')
    assert_equal(a[0]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.r_[1, -1, 2, -2, 3])
    assert_equal(a[1]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.array([[-1, 1], [-2, 2]]))


def test_2out_label():
    a = X('[a0, b0, a1, b1, ...]; [[b0, a0], [b1, a1], ...]')
    assert_equal(a[0]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.r_[1, -1, 2, -2, 3])
    assert_equal(a[1]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.array([[-1, 1], [-2, 2]]))


def test_2out_list():
    a = X('[1, 2, ...]; [a, b, ...]', ['[1, a, 2, b, ...]', '[[a, 1], [b, 2], ...]'])
    assert_equal(a[0]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.r_[1, -1, 2, -2, 3])
    assert_equal(a[1]([np.r_[1, 2, 3, 4], np.r_[-1, -2]]), np.array([[-1, 1], [-2, 2]]))


def test_func():
    assert_equal(X('[1, ...]', '[fun(1), ...]', f={'fun': lambda x: x * 10})(np.r_[1, 2, 3]), np.r_[10, 20, 30])


def test_func2():
    assert_equal(X('[1, ...]', '[fun(1), ...]')(np.r_[1, 2, 3], f={'fun': lambda x: x * 10}), np.r_[10, 20, 30])
    assert_equal(X('[1, ...]', '[fun(1), ...]')(np.r_[1, 2, 3], f={'fun': lambda x: x * 5}), np.r_[5, 10, 15])


def test_func3():
    assert_equal(X('[[1, 2], ...]', '[[fun(1), 2], ...]', f={'fun': lambda x: x * 10})(np.array([[1, 2], [3, 4]])),
                 np.array([[10, 2], [30, 4]]))


def test_func3_label():
    assert_equal(X('[[fun(a00), a01], ...]', f={'fun': lambda x: x * 10})(np.array([[1, 2], [3, 4]])),
                 np.array([[10, 2], [30, 4]]))


def test_outShape():
    assert_equal(X('[1, ...]', '[[1, 1, ...], ...]')(np.r_[1, 2, 3], outShapes=(-1, 2)),
                 np.array([[1, 1], [2, 2], [3, 3]]))


def test_outShape2():
    assert_equal(X('[1, ...]', '[[1, 1, ...], ...]')(np.r_[1, 2, 3], outShapes=[(-1, 2)]),
                 np.array([[1, 1], [2, 2], [3, 3]]))


def test_outShape3():
    assert_equal(X('[[1, 2], ...]', '[[1, 2], ...]')(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]]))


def test_outShape3():
    assert_equal(X('[[a00, a01], ...]')(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]]))


def test_extraShape():
    assert_equal(X('[1, 2, 3, ...]', '[[1, 2], [2, 3], ...]')(np.r_[0, 1, 2, 3], extraShapes=(1, 0)),
                 np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))


def test_extraShape2():
    assert_equal(
        X('[1, 2, 3, ...]', '[[1, 2], [2, 3], ...]; [[1, 2, 3], ...]')(np.r_[0, 1, 2, 3], extraShapes=[(1, 0), (1, 0)]),
        (np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]])))
