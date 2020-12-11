from shapely.geometry import LineString
from matplotlib.pyplot import *
from numpy import *
from dask.array.creation import linspace
from numpy.polynomial.polynomial import polyline
from networkx.generators import intersection
       
# Given a set of points with coordinates x and y, returns the line[x,y]
def points2polyline(x,y):
    return np.vstack((x, y)).T

def polyline2points(polyline):
    return polyline[:,0],polyline[:,1]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def poly_intersection(poly1, poly2):

    xc=[]
    yc=[]
    for i, p1_first_point in enumerate(poly1[:-1]):
        p1_second_point = poly1[i + 1]

        for j, p2_first_point in enumerate(poly2[:-1]):
            p2_second_point = poly2[j + 1]

            x,y = line_intersection((p1_first_point, p1_second_point), (p2_first_point, p2_second_point))
            if x != None:
                xc.append(x)
                yc.append(y)
                
    return xc,yc

    return False

start_time = time.time()
x = arange(0,10,0.1)
y1 = sin(x);
y2 = cos(x);

ion()
 
# PL1 = ((-1, -1), (1, -1), (1, 2))
# PL2 = ((0, 1), (2, 1))
PL1 = points2polyline(x, y1)
PL2 = points2polyline(x, y2)

xc,yc = poly_intersection(PL1, PL2)
print("start_time %s seconds" % (time.time() - start_time))
plot(x,y1,x,y2)
plot(xc,yc,'ro')
print ("end")