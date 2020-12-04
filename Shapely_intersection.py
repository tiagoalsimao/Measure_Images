from shapely.geometry import LineString
from shapely.geometry import Point
# from shapely.geometry import *
from matplotlib.pyplot import *
from numpy import *
from dask.array.creation import linspace
from numpy.polynomial.polynomial import polyline
from networkx.generators import intersection

def intersections2coords(intersectionMultiPoint):
    
    # There is no intersection
    if type(intersectionMultiPoint) is LineString:
        return [], []
    
    # check intersection is a single point
    if type(intersectionMultiPoint) is Point:
        return intersectionMultiPoint.coords.xy
    
    # return multi points of intersection
    nPoints = len(intersectionMultiPoint)
    xi = np.zeros(nPoints)
    yi = np.zeros(nPoints)
    for i in range(nPoints):
        xi[i] = intersectionMultiPoint[i].x
        yi[i] = intersectionMultiPoint[i].y
    
    return xi,yi

# Given a set of points with coordinates x and y, returns the line[x,y]
def coords2polyline(x,y):
    return c_[x, y]

def polyline2coords(polyline):
    return polyline[:,0],polyline[:,1]

start_time = time.time()
x = arange(0,3,0.1)
y1 = sin(x);
y2 = cos(x);

ion()
plot(x,y1,x,y2)

PL1 = coords2polyline(x, y1)
PL2 = coords2polyline(x, y2)

line1 = LineString(PL1)
line2 = LineString(PL2)

lx1,ly1 = line1.coords.xy
lx2,ly2 = line2.coords.xy
xi,yi = intersections2coords(line1.intersection(line2))

print("start_time %s seconds" % (time.time() - start_time))
plot(lx1,ly1,lx2,ly2)
plot(xi,yi,'ro')

print ("end")