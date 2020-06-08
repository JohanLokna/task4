import numpy as np
from siameese import getTriplets

top=500
distFile = 'distResnet1000_1.txt'
resultFile = 'resultsResnet1000_1.txt'

bothDist = np.genfromtxt(distFile, delimiter=' ')
dists = np.abs(bothDist[:, 0] - bothDist[:, 1])
indeciesLow = dists.argsort()[:-top][::-1]
indeciesHigh = dists.argsort()[-top:][::-1]

results = np.genfromtxt(resultFile, dtype=int)

with open('top500Resnet_1.txt', 'w+') as f:
  allTriplets = getTriplets('test_triplets.txt')
  for i in indeciesHigh:

    if results[i] == 0:
      allTriplets[i][1], allTriplets[i][2] = allTriplets[i][2], allTriplets[i][1]

    f.write(' '.join(allTriplets[i]) + '\n')

results[indeciesLow] = 1 - results[indeciesLow]
np.savetxt('negResultsResnet1000_1_500.txt', results, fmt='%i')
