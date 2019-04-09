b = T_Alert # get the 'Failure Alert' labels (target point in output space, 6600x1)
bp = np.dot(Xa, w) # get the point in output space, closest to 'b' (minimizes error), that lies on the shaded plane (6600x1 as well)
# NOTE: 'b' is composed of +1/-1, while 'bp' is composed of +ve/-ve values (i.e. 'bp' is an approximation of 'b'). 'bp' is used to classify the 6600 samples as +1 / -1.

dist_bp = np.sqrt(np.sum(np.square(b-bp))) # compute distance between the two points

# get the min,max values in weight vector (used to create some random weight vectors below)
lWeight = np.amin(w) 
hWeight = np.amax(w)

nTries = 10000 # number of random tries
randomBs = []
distances = []
for x in range(nTries):
    wTry = np.random.uniform(lWeight, hWeight, len(w)) # get a random weight vector (16x1)
    bTry = np.dot(Xa, wTry) # get corresponding point in output space (6600x1), that lies on shaded plane
    dist_b = np.sqrt(np.sum(np.square(b-bTry))) # compute distance between target point and this random point
    print(dist_b)
    if (dist_b<1000): # if we got a close enough point, keep track of it for reporting
        randomBs.append(bTry)
        distances.append(dist_b)