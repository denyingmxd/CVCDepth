import numpy as np
import matplotlib.pyplot as plt

names=['ours','VFDepth','SurroundDepth']
# rels_ddad = [0.204,0.198,0.218,0.208]
# rels_nusc = [0.230,0.225,0.289,0.280]
# memories_ddad = [7138, 7754, 16297, 20131]
# memories_nusc = [6536, 7362, 15629, 18201]

names=['ours','VFDepth','SurroundDepth']
rels_ddad = np.array([0.198,0.218,0.208])
rels_nusc = np.array([0.225,0.289,0.280])
memories_ddad =  np.array([5780,7400, 7138, 7754, 16297, 20131])
memories_nusc =  np.array([6178,6950,6536, 7362, 15629, 18201])
# memories_ddad = np.array([7754, 16297, 20131])
# memories_nusc = np.array([7362, 15629, 18201])opo
print(memories_ddad/1024)
print(memories_nusc/1024)
exit()

import matplotlib.pyplot as plt
import pandas as pd


df =pd.DataFrame({"amount": rels_ddad, "price": memories_ddad})

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.2

df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_xticklabels(names, rotation='horizontal', fontsize=12)
ax.set_ylabel('Amount')
ax2.set_ylabel('Price')



plt.tight_layout()


plt.show()



plt.show()