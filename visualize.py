import numpy as np
import matplotlib.pyplot as plt

def draw(path):
    with open(path,'r') as f:
        lines = f.readlines()

    d = dict()

    for line in lines:
        sps = line.split()
        label = int(sps[0])
        if label not in d:
            d[label] = []
        pt = [float(sps[1]), float(sps[2])]
        d[label].append(pt)
    print len(d)
    color = ['blue','green','red','cyan','magenta',
        'yellow','black','white','gray','pink']
    for key in d:
        print str(key) + "\t" + str(len(d[key]))
        val = np.array(d[key])
        x = val[:,0]
        y = val[:,1]
        plt.scatter(x,y, label=key, c = color[key])

    plt.legend()
    plt.show()

draw('center_feat.log')
