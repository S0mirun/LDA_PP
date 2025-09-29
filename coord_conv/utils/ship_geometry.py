
import numpy as np

import coord_conv


def R(p):
    R = np.array([
        [ np.cos(p[2]), -np.sin(p[2]), 0.0 ],
        [ np.sin(p[2]), np.cos(p[2]), 0.0 ],
        [ 0.0, 0.0, 1.0 ],
    ])
    return R

def transform_EF(
        x_oEF, y_oEF, psi_oEF,
        Ox_oEF, Oy_oEF, theta_oEF,
):
    """
    transform x,y coordinate and heading angle
    on the original Earth-fixed frame
    into values on new Earth-fexed frame

    Args:
        x_oEF (_type_):
            x coordnate of given ship position on the original Earth-fixed frame
        y_oEF (_type_):
            y coordnate of given ship position on the original Earth-fixed frame
        psi_oEF (_type_):
            heading angle of given ship on the original Earth-fixed frame
        Ox_oEF (_type_):
            x coordinate of the origin of new Earth-fixed frame
            on the original Earth-fixed frame
        Oy_oEF (_type_):
            y coordinate of the origin of new Earth-fixed frame
            on the original Earth-fixed frame
        theta_oEF (_type_):
            argument angle of new Earth-fixed frame
            on the original Earth-fixed frame

    Returns:
        _type_:
            x coordnate, y coordnate, and heading angle of given ship
            on new Earth-fixed frame
    """
    p_oEF = np.array([x_oEF, y_oEF, psi_oEF])
    p_origin_oEF = np.array([Ox_oEF, Oy_oEF, theta_oEF])
    p_nEF = np.transpose(R(p_origin_oEF)) @ (p_oEF - p_origin_oEF)
    return p_nEF

def p_dot_EF(p_EF, v_BF):
    p_dot = R(p_EF) @ v_BF
    return p_dot

def clip_angle(psi):
    return np.arctan2( np.sin(psi), np.cos(psi) )

def ship_shape_poly(pose, L=3.0, B=0.48925, scale=1, Lrate = 0.6, z_axis_upward=False):
    # set state
    x, y, p = pose
    xc  = x if z_axis_upward else y
    yc  = y if z_axis_upward else x
    psi = p if z_axis_upward else np.pi*0.5 - p
    # calc shape
    coo1 = np.array([xc + scale * L/2.0 * np.cos(psi), yc + scale * L/2.0 * np.sin(psi)])
    coo2 = np.array([xc + scale * (L/2.0 * Lrate * np.cos(psi) - B/2.0 * np.sin(psi)), yc + scale * (L/2.0 * Lrate * np.sin(psi) + B/2.0 * np.cos(psi))])
    coo3 = np.array([xc + scale * ((-1) * L/2.0 * np.cos(psi) - B/2.0 * np.sin(psi)), yc + scale * ((-1) * L/2.0 * np.sin(psi) + B/2.0 * np.cos(psi))])
    coo4 = np.array([xc + scale * ((-1) * L/2.0 * np.cos(psi) + B/2.0 * np.sin(psi)), yc + scale * ((-1) * L/2.0 * np.sin(psi) - B/2.0 * np.cos(psi))])
    coo5 = np.array([xc + scale * (L/2.0 * Lrate * np.cos(psi) + B/2.0 * np.sin(psi)), yc + scale * (L/2.0 * Lrate * np.sin(psi) - B/2.0 * np.cos(psi))])
    return (coo1, coo2, coo3, coo4, coo5)

def convert_to_xy(lat, lon, lat_origin, lon_origin, angle_from_north):
    longitude_array = np.array([lon_origin, lon])
    latitude_array = np.array([lat_origin, lat])
    distance_min, course_to_north = coord_conv.lat_lon.caldistco(longitude_array, latitude_array)
    distance_meter = distance_min * 1852.0
    course_radian = (course_to_north - angle_from_north) * np.pi / 180
    x_point, y_point = distance_meter * np.cos(course_radian), distance_meter * np.sin(course_radian)
    return x_point, y_point

def knot_to_ms(speed_knot):
    return speed_knot * 1852.0 / 3600.0