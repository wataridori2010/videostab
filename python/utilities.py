#!/usr/bin/env python

import numpy as np
import math
import cv2

def get_gaussian_kernel(sigma2, v1, v2, normalize=True):
    gauss = [math.exp(-(float(x*x) / sigma2)) for x in range(v1, v2+1)]
    total = sum(gauss)

    if normalize:
        gauss = [x/total for x in gauss]

    return gauss
    

def gaussian_filter(input_array):
    """
    """
    # Step 1: Define the convolution kernel
    sigma = 10000
    r = 256
    kernel = get_gaussian_kernel(4000, -r, r)

    # Step 2: Convolve
    return np.convolve(input_array, kernel, 'same')

def diff(timestamps):
    """
    Returns differences between consecutive elements
    """
    return np.ediff1d(timestamps)

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out

    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """
    Generic method to return a rotation matrix
    - +ve angle = anticlockwise
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])

    R  = np.diag([cosa, cosa, cosa])
    R +=  np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([ [0.0,           -direction[2],  direction[1]],
                    [direction[2],  0.0,           -direction[0]],
                    [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)

    return M

def rotx_matrix(angle):
    return rotation_matrix(angle, [1, 0, 0])

def roty_matrix(angle):
    return rotation_matrix(angle, [0, 1, 0])

def rotz_matrix(angle):
    return rotation_matrix(angle, [0, 0, 1])

def meshwarp(src, distorted_grid):
    """
    src: The source image (Mat)
    distorted_grid: An n*n distorted_grid 
    """
    size = src.shape
    mapsize = (size[0], size[1], 1)
    dst = np.zeros(size, dtype=np.uint8)
    #mapx = np.zeros(mapsize, dtype=np.float32)
    #mapy = np.zeros(mapsize, dtype=np.float32)

    quads_per_row = len(distorted_grid[0]) - 1
    quads_per_col = len(distorted_grid) - 1
#    pixels_per_row = size[1] / quads_per_row
#    pixels_per_col = size[0] / quads_per_col
    pixels_per_row = size[1] / (quads_per_row+1)
    pixels_per_col = size[0] / (quads_per_col+1)

    print "pixels_per_row"    
    print pixels_per_row
    print "pixels_per_col"    
    print pixels_per_col

    
    pt_src_all = []
    pt_src_all2 = []    
    pt_dst_all = []

    for i in distorted_grid:
        pt_src_all.extend(i)
        
    print pt_src_all[0]
    abc=pt_src_all[0]
    abcd=abc[0]
    abcde=abcd[0]    
    print abcde[0]
    print abcde[1]    
        
    print "quads_per_row"    
    print quads_per_row
    print "quads_per_col"    
    print quads_per_col
    
    tcount=0
    
    for x in range(quads_per_row+1):
        for y in range(quads_per_col+1):
#            pt_dst_all.append([x*pixels_per_col, y*pixels_per_row])
            pt_dst_all.append([x*pixels_per_row, y*pixels_per_col])
    
            abc=pt_src_all[tcount]
            abcd=abc[0]
            abcde=abcd[0]    
            pt_src_all2.append([abcde[0],abcde[1]])
            tcount=tcount+1

    
    gx, gy = np.mgrid[0:size[1], 0:size[0]]
    
    print "gx"    
    print gx
#    print gx.shape
    print "gy"    
    print gy
#    print gy.shape    
#    print "pt_dst_all"
#    print np.array(pt_dst_all).shape
#    print "pt_src_all"    
    print pt_src_all2
    print pt_dst_all
    
    import scipy
    from scipy.interpolate import griddata

#    g_out = griddata(np.array(pt_dst_all), np.array(pt_src_all), (gx, gy), method='linear')
    g_out = griddata(np.array(pt_dst_all), np.array(pt_src_all2), (gx, gy), method='linear')

    import matplotlib.pyplot as plt
#    plt.plot(gx, gy) 
    print "g_out"    
    print g_out
    print "g_out_size"    
    print g_out.size    
#    print "mapsize"    
#    print mapsize

#    mapx = np.append([], [ar[:,0] for ar in g_out]).reshape(mapsize).astype('float32')
#    mapy = np.append([], [ar[:,1] for ar in g_out]).reshape(mapsize).astype('float32')
#    mapx = np.append([], [ar[:,:,:,0] for ar in g_out]).reshape(mapsize).astype('float32')# ito
#    mapy = np.append([], [ar[:,:,:,1] for ar in g_out]).reshape(mapsize).astype('float32')# ito

#    mx = np.append([], [ar[:,:,:,0] for ar in g_out])
#    my = np.append([], [ar[:,:,:,1] for ar in g_out])
    mx = np.append([], [ar[:,0] for ar in np.array(g_out)])#ito
    my = np.append([], [ar[:,1] for ar in np.array(g_out)])#ito

    print "mx"
    print mx.shape
    
#    import pandas as pd
#    import math
#    for i in range(mx.size-1):
#        if math.isnan(mx[i]):
#            mx[i]=0
#        if math.isnan(my[i]):
#            my[i]=0
    
    print mx[0:6]
    print mx[543:549]
    
    print "my"    
    print my
    
#    mx=np.array(mx)
#    my=np.array(my)
#    print "mx"    
#    print mx[0]
#    print mx[1]
#    print mx[2]        
#    mapx = mx.reshape(mapsize).astype('float32')# ito
#    mapy = my.reshape(mapsize).astype('float32')# ito    
    mapx = mx.reshape((mapsize[1],mapsize[0],1)).astype('float32')# ito
    mapy = my.reshape((mapsize[1],mapsize[0],1)).astype('float32')# ito    
#    mapx = mx.reshape((mapsize[0],mapsize[1],1)).astype('float32')# ito
#    mapy = my.reshape((mapsize[0],mapsize[1],1)).astype('float32')# ito    
    mapx = mapx.transpose((1,0,2))
    mapy = mapy.transpose((1,0,2))
    
    print "mapx.size"    
    print mapx.shape
    
    
    print mapx[0:6,0,0]
    print mapx[543,0,0]
    print mapx[0:6,1,0]
    
    print "(mapsize[0],mapsize[1])"
    print (mapsize[0],mapsize[1])
    

    print "src.size"    
    print src.shape

    print mapx
    print mapy    
#    src1 = src[:,:,2].copy()
#    print "src1.size"    
#    print src1.shape
#    cv2.imwrite('src1.png', src1)
    cv2.imwrite('src.png', src)
    cv2.imwrite('mapx.png', mapx*10)
    cv2.imwrite('mapy.png', mapy*10)

#    for x in range(mapsize[1]-1):
#        for y in range(mapsize[0]-1):
#            print "%d, %d, %f" % (x,y,mapy[y,x])
    
#    exit(1)

#    for x in range(mapsize[1]-1):
#        for y in range(mapsize[0]-1):
#            mapy[y,x]=y
#            mapx[y,x]=x            
    
    dst = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imwrite('test.png', dst)
    return dst

if __name__ == '__main__':
    #rx = rotx_matrix(math.radians(45))
    #ry = roty_matrix(math.radians(45))
    #rz = rotz_matrix(math.radians(45))

    #print rz * rx * ry
    img = cv2.imread("/work/blueprints/c7/data/mesh2.png")

    grid = [[(0,0), (100, 0), (200, 0), (300, 0)],
            [(0, 100), (150, 50), (200, 50), (300, 100)],
            [(0, 200), (150, 100), (200, 100), (300, 200)],
            [(0, 300), (100, 300), (200, 300), (300, 300)],
    ]

    meshwarp(img, grid)

