import numpy as np
cimport numpy as cnp # コンパイル（コツ1）
cimport cython       # コンパイル（コツ1）
from cython import boundscheck, wraparound  # 配列チェック機能（コツ7）

ctypedef cnp.float64_t DTYPE_t  # numpyの詳細な型指定（コツ3）


cpdef inline bint cy_intersect(cnp.ndarray p1, cnp.ndarray p2, cnp.ndarray p3, cnp.ndarray p4):
    cdef double tc1
    cdef double tc2
    cdef double td1
    cdef double td2

    with  boundscheck(False), wraparound(False):
        tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1*tc2<0 and td1*td2<0

cpdef inline cnp.ndarray cy_line_cross_point(cnp.ndarray p1, cnp.ndarray p2, cnp.ndarray p3, cnp.ndarray p4):
    cdef double a0
    cdef double b0
    cdef double a2
    cdef double b2
    cdef double d
    cdef double sn

    with  boundscheck(False), wraparound(False):
        a0 = p2[0] - p1[0]
        b0 = p2[1] - p1[1]
        a2 = p4[0] - p3[0]
        b2 = p4[1] - p3[1]
        d = a0*b2 - a2*b0
        if d == 0:
            return None
        sn = b2 * (p3[0]-p1[0]) - a2 * (p3[1]-p1[1])
        return np.array([p1[0] + a0*sn/d, p1[1] + b0*sn/d])

cpdef inline cnp.ndarray cy_DDA_2D(cnp.ndarray start, cnp.ndarray end, double pitch, int n_loops):
    cdef cnp.ndarray[DTYPE_t, ndim=1] start_node  # numpyの詳細な型指定（コツ3）
    cdef cnp.ndarray[DTYPE_t, ndim=1] end_node
    cdef cnp.ndarray[DTYPE_t, ndim=1] direction_vector
    cdef double step_x
    cdef double step_y
    cdef double tx
    cdef double ty
    cdef cnp.ndarray[DTYPE_t, ndim=2] current_node
    cdef cnp.ndarray[DTYPE_t, ndim=2] desired_node_list


    start_node =np.array([pitch * np.floor(start[0]/pitch), pitch * np.floor(start[1]/pitch)])
    end_node =np.array([pitch * np.floor(end[0]/pitch), pitch * np.floor(end[1]/pitch)])
    direction_vector = end - start
    step_x = np.sign(direction_vector[0])
    step_y = np.sign(direction_vector[1])

    if direction_vector[0] > 0:
        tx = (start_node[0] + pitch - start[0]) / abs(direction_vector[0])
    elif direction_vector[0] < 0:
        tx = (start[0] - start_node[0]) / abs(direction_vector[0])
    else:
        tx = np.inf

    if direction_vector[1] > 0:
        ty = (start_node[1] + pitch - start[1]) / abs(direction_vector[1])
    elif direction_vector[1] < 0:
        ty = (start[1] - start_node[1]) / abs(direction_vector[1])
    else:
        ty = np.inf

    current_node = np.array([start_node])
    desired_node_list = np.empty((n_loops, 2))
    for i in range(n_loops):
        desired_node_list[i] = current_node
        if np.array_equal(current_node[0,:], end_node):
            desired_node_list = desired_node_list[:i+1]
            break
        if tx < ty:
            tx += pitch / abs(direction_vector[0])
            current_node[0,0] += step_x * pitch
        elif ty < tx:
            ty += pitch /abs(direction_vector[1])
            current_node[0,1] += step_y * pitch
        else:
            tx += pitch / abs(direction_vector[0])
            ty += pitch /abs(direction_vector[1])
            current_node[0,0] += step_x * pitch
            current_node[0,1] += step_y * pitch

    return desired_node_list