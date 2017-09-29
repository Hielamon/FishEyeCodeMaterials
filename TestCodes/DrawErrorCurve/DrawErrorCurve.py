import matplotlib.pyplot as plt
import numpy as np


generalModelName = ["PolynomialAngle","PolynomialRadius","GeyerModel"]
curveStyle = ["r-", "b-", "g-"]
errorPath = []
fig = plt.figure()
idx = 0
for name in generalModelName:
    nameTmp = "../OptimizeMetric/" + name + "_error.txt"
    errorPath.append(nameTmp)
    file = open(nameTmp)
    iters = []
    errors = []
    for line in file.readlines():
        line = line.replace("\n", "").split(" ")
        iters.append(int(line[0]))
        errors.append(float(line[1]))
    plt.plot(iters, errors, curveStyle[idx], label = name)

    idx += 1

plt.legend()
plt.grid()
plt.title("The Error Curve of Diff General FishEye Model")
plt.xlabel(r"iter")
plt.ylabel(r"error")
plt.show()

