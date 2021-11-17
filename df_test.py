import pandas as pd

# Define a dictionary containing data
data = {'a': [0,0,0.671399,0.446172,0,0.614758, ],
    'b': [0,0,0.101208,-0.243316,0,0.075793],
    'c': [0,0,-0.181532,0.051767,0,-0.451460],
    'd': [1,0,1,1.577318,1,-0.012493]}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

# print(df)

# assign few value to new column
# df["new"] = [1,12]

# print(list(df[df["d"]==1].index.values))

l = [[(1,1), (2,2)], [(2,2),(4,5)], [(4,4)]]
import itertools
# flatten a list of lists
l = list(itertools.chain(*l))

print(l.index, l.values)
#