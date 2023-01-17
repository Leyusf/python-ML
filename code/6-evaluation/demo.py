from refression_evaluator import RegressionEvaluator
import numpy as np

r = RegressionEvaluator("rmse")
Y = np.array([1, 2, 3, 4, 5])
Y_p = np.array([2, 2, 3, 1, 4])
print(r.evaluate(Y, Y_p))
