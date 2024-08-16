import pandas as pd
import matplotlib.pyplot as plt


evaluation = pd.read_csv('../report.csv')
evaluation_results = evaluation.groupby(['Frame']).sum()
print(evaluation_results)

evaluation_results.plot.bar()
plt.show()
