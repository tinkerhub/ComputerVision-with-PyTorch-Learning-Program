from sklearn.linear_model import LinearRegression
import pytest
x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
model = LinearRegression()

model.fit(x, y)

linear_test_set = [([[10]], 12), ([[11]], 13)]

@pytest.mark.parametrize(("ip", "op"), linear_test_set)
def test_code(ip, op):
    assert int(model.predict(ip)[0]) == op


