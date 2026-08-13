# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

# This is the pure-Python fallback for core.math_helpers.
# When the Cython extension is built (math_helpers.pyx -> .pyd),
# it shadows this file at import time for much better performance.
from __future__ import annotations
import numpy as np
import math
FLOAT_TYPE = np.float64


def mat4_mul_fast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros((4, 4), dtype=FLOAT_TYPE)
    for i in range(4):
        a0,a1,a2,a3 = a[i,0],a[i,1],a[i,2],a[i,3]
        r[i,0] = a0*b[0,0]+a1*b[1,0]+a2*b[2,0]+a3*b[3,0]
        r[i,1] = a0*b[0,1]+a1*b[1,1]+a2*b[2,1]+a3*b[3,1]
        r[i,2] = a0*b[0,2]+a1*b[1,2]+a2*b[2,2]+a3*b[3,2]
        r[i,3] = a0*b[0,3]+a1*b[1,3]+a2*b[2,3]+a3*b[3,3]
    return r


def mat4_inv_fast(m: np.ndarray) -> np.ndarray:
    m00,m01,m02,m03 = m[0,0],m[0,1],m[0,2],m[0,3]
    m10,m11,m12,m13 = m[1,0],m[1,1],m[1,2],m[1,3]
    m20,m21,m22,m23 = m[2,0],m[2,1],m[2,2],m[2,3]
    m30,m31,m32,m33 = m[3,0],m[3,1],m[3,2],m[3,3]
    t00 = m11*m22*m33-m11*m23*m32-m12*m21*m33+m12*m23*m31+m13*m21*m32-m13*m22*m31
    t10 = -m10*m22*m33+m10*m23*m32+m12*m20*m33-m12*m23*m30-m13*m20*m32+m13*m22*m30
    t20 = m10*m21*m33-m10*m23*m31-m11*m20*m33+m11*m23*m30+m13*m20*m31-m13*m21*m30
    t30 = -m10*m21*m32+m10*m22*m31+m11*m20*m32-m11*m22*m30-m12*m20*m31+m12*m21*m30
    det = m00*t00+m01*t10+m02*t20+m03*t30
    if abs(det) < 1e-15:
        return np.eye(4, dtype=FLOAT_TYPE)
    inv_det = 1.0/det
    inv = np.empty((4,4), dtype=FLOAT_TYPE)
    inv[0,0] = t00*inv_det
    inv[1,0] = t10*inv_det
    inv[2,0] = t20*inv_det
    inv[3,0] = t30*inv_det
    inv[0,1] = (-m01*m22*m33+m01*m23*m32+m02*m21*m33-m02*m23*m31-m03*m21*m32+m03*m22*m31)*inv_det
    inv[1,1] = (m00*m22*m33-m00*m23*m32-m02*m20*m33+m02*m23*m30+m03*m20*m32-m03*m22*m30)*inv_det
    inv[2,1] = (-m00*m21*m33+m00*m23*m31+m01*m20*m33-m01*m23*m30-m03*m20*m31+m03*m21*m30)*inv_det
    inv[3,1] = (m00*m21*m32-m00*m22*m31-m01*m20*m32+m01*m22*m30+m02*m20*m31-m02*m21*m30)*inv_det
    inv[0,2] = (m01*m12*m33-m01*m13*m32-m02*m11*m33+m02*m13*m31+m03*m11*m32-m03*m12*m31)*inv_det
    inv[1,2] = (-m00*m12*m33+m00*m13*m32+m02*m10*m33-m02*m13*m30-m03*m10*m32+m03*m12*m30)*inv_det
    inv[2,2] = (m00*m11*m33-m00*m13*m31-m01*m10*m33+m01*m13*m30+m03*m10*m31-m03*m11*m30)*inv_det
    inv[3,2] = (-m00*m11*m32+m00*m12*m31+m01*m10*m32-m01*m12*m30-m02*m10*m31+m02*m11*m30)*inv_det
    inv[0,3] = (-m01*m12*m23+m01*m13*m22+m02*m11*m23-m02*m13*m21-m03*m11*m22+m03*m12*m21)*inv_det
    inv[1,3] = (m00*m12*m23-m00*m13*m22-m02*m10*m23+m02*m13*m20+m03*m10*m22-m03*m12*m20)*inv_det
    inv[2,3] = (-m00*m11*m23+m00*m13*m21+m01*m10*m23-m01*m13*m20-m03*m10*m21+m03*m11*m20)*inv_det
    inv[3,3] = (m00*m11*m22-m00*m12*m21-m01*m10*m22+m01*m12*m20+m02*m10*m21-m02*m11*m20)*inv_det
    return inv


def mat4_translation(pos_x: float, pos_y: float, pos_z: float) -> np.ndarray:
    m = np.eye(4, dtype=FLOAT_TYPE)
    m[3,0] = pos_x; m[3,1] = pos_y; m[3,2] = pos_z
    return m


def mat4_scale_mat(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=FLOAT_TYPE)
    m[0,0] = sx; m[1,1] = sy; m[2,2] = sz
    return m


def mat4_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    m = np.eye(4, dtype=FLOAT_TYPE)
    x2,y2,z2 = x+x,y+y,z+z
    xx,xy,xz = x*x2,x*y2,x*z2
    yy,yz,zz = y*y2,y*z2,z*z2
    wx,wy,wz = w*x2,w*y2,w*z2
    m[0,0]=1.0-(yy+zz); m[0,1]=xy+wz;      m[0,2]=xz-wy
    m[1,0]=xy-wz;       m[1,1]=1.0-(xx+zz); m[1,2]=yz+wx
    m[2,0]=xz+wy;       m[2,1]=yz-wx;       m[2,2]=1.0-(xx+yy)
    return m


def quat_conjugate(x: float, y: float, z: float, w: float):
    return -x, -y, -z, w


def quat_mul(ax: float, ay: float, az: float, aw: float, bx: float, by: float, bz: float, bw: float):
    return (
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
        aw*bw-ax*bx-ay*by-az*bz
    )


def quat_slerp(ax: float, ay: float, az: float, aw: float, bx: float, by: float, bz: float, bw: float, t: float):
    dot = ax*bx+ay*by+az*bz+aw*bw
    if dot < 0.0:
        bx=-bx; by=-by; bz=-bz; bw=-bw; dot=-dot
    if dot > 0.9995:
        rx=ax+t*(bx-ax); ry=ay+t*(by-ay); rz=az+t*(bz-az); rw=aw+t*(bw-aw)
        n=(rx*rx+ry*ry+rz*rz+rw*rw)**0.5
        if n > 1e-10:
            inv_n=1.0/n
            return rx*inv_n,ry*inv_n,rz*inv_n,rw*inv_n
        return ax,ay,az,aw
    theta0=math.acos(dot)
    sin_theta0=math.sin(theta0)
    inv_sin=1.0/sin_theta0
    theta=theta0*t
    s0=(math.cos(theta)-dot*math.sin(theta))*inv_sin
    s1=math.sin(theta)*inv_sin
    rx=s0*ax+s1*bx; ry=s0*ay+s1*by; rz=s0*az+s1*bz; rw=s0*aw+s1*bw
    n=(rx*rx+ry*ry+rz*rz+rw*rw)**0.5
    if n > 1e-10:
        inv_n=1.0/n
        return rx*inv_n,ry*inv_n,rz*inv_n,rw*inv_n
    return ax,ay,az,aw


def quat_from_euler(x_deg: float, y_deg: float, z_deg: float):
    hx=math.radians(x_deg)*0.5
    hy=math.radians(y_deg)*0.5
    hz=math.radians(z_deg)*0.5
    sx,cx=math.sin(hx),math.cos(hx)
    sy,cy=math.sin(hy),math.cos(hy)
    sz,cz=math.sin(hz),math.cos(hz)
    return (sx*cy*cz-cx*sy*sz,
            cx*sy*cz+sx*cy*sz,
            cx*cy*sz-sx*sy*cz,
            cx*cy*cz+sx*sy*sz)


def quat_to_euler(x: float, y: float, z: float, w: float):
    sinx_cosp=2*(w*x+y*z)
    cosx_cosp=1-2*(x*x+y*y)
    rx=math.degrees(math.atan2(sinx_cosp,cosx_cosp))
    siny_cosp=2*(w*y-z*x)
    ry=math.degrees(math.asin(max(-1.0,min(1.0,siny_cosp))))
    sinz_cosp=2*(w*z+x*y)
    cosz_cosp=1-2*(y*y+z*z)
    rz=math.degrees(math.atan2(sinz_cosp,cosz_cosp))
    return rx,ry,rz


def quat_from_euler_rad(x_rad: float, y_rad: float, z_rad: float):
    hx=x_rad*0.5; hy=y_rad*0.5; hz=z_rad*0.5
    sx,cx=math.sin(hx),math.cos(hx)
    sy,cy=math.sin(hy),math.cos(hy)
    sz,cz=math.sin(hz),math.cos(hz)
    return (sx*cy*cz-cx*sy*sz,
            cx*sy*cz+sx*cy*sz,
            cx*cy*sz-sx*sy*cz,
            cx*cy*cz+sx*sy*sz)


def quat_to_euler_rad(x: float, y: float, z: float, w: float):
    sinx_cosp=2*(w*x+y*z); cosx_cosp=1-2*(x*x+y*y)
    rx=math.atan2(sinx_cosp,cosx_cosp)
    siny_cosp=2*(w*y-z*x)
    ry=math.asin(max(-1.0,min(1.0,siny_cosp)))
    sinz_cosp=2*(w*z+x*y); cosz_cosp=1-2*(y*y+z*z)
    rz=math.atan2(sinz_cosp,cosz_cosp)
    return rx,ry,rz


def quat_normalize(x: float, y: float, z: float, w: float):
    n=(x*x+y*y+z*z+w*w)**0.5
    if n > 1e-10:
        inv_n=1.0/n
        return x*inv_n,y*inv_n,z*inv_n,w*inv_n
    return 0.0,0.0,0.0,1.0


def quat_rotate_vec3(qx: float, qy: float, qz: float, qw: float, vx: float, vy: float, vz: float):
    tx=2.0*(qy*vz-qz*vy)
    ty=2.0*(qz*vx-qx*vz)
    tz=2.0*(qx*vy-qy*vx)
    return (
        vx+qw*tx+qy*tz-qz*ty,
        vy+qw*ty+qz*tx-qx*tz,
        vz+qw*tz+qx*ty-qy*tx
    )


def ray_sphere_intersect_batch(
    origins_x: np.ndarray, origins_y: np.ndarray, origins_z: np.ndarray,
    dirs_x: np.ndarray, dirs_y: np.ndarray, dirs_z: np.ndarray,
    centers_x: np.ndarray, centers_y: np.ndarray, centers_z: np.ndarray,
    radii: np.ndarray,
    results: np.ndarray
):
    for i in range(len(radii)):
        ocx=origins_x[i]-centers_x[i]
        ocy=origins_y[i]-centers_y[i]
        ocz=origins_z[i]-centers_z[i]
        dx,dy,dz=dirs_x[i],dirs_y[i],dirs_z[i]
        b=ocx*dx+ocy*dy+ocz*dz
        c=ocx*ocx+ocy*ocy+ocz*ocz-radii[i]*radii[i]
        disc=b*b-c
        if disc < 0.0:
            results[i]=-1.0
            continue
        sq=disc**0.5
        t=-b-sq
        if t > 0.0:
            results[i]=t
        else:
            t2=-b+sq
            results[i]=t2 if t2 > 0.0 else -1.0


def mat3x3_inv(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    det=m00*(m11*m22-m12*m21)-m01*(m10*m22-m12*m20)+m02*(m10*m21-m11*m20)
    if abs(det) < 1e-15:
        return np.eye(3, dtype=FLOAT_TYPE)
    inv_det=1.0/det
    r=np.empty((3,3), dtype=FLOAT_TYPE)
    r[0,0]=(m11*m22-m12*m21)*inv_det; r[0,1]=-(m01*m22-m02*m21)*inv_det; r[0,2]=(m01*m12-m02*m11)*inv_det
    r[1,0]=-(m10*m22-m12*m20)*inv_det; r[1,1]=(m00*m22-m02*m20)*inv_det; r[1,2]=-(m00*m12-m02*m10)*inv_det
    r[2,0]=(m10*m21-m11*m20)*inv_det; r[2,1]=-(m00*m21-m01*m20)*inv_det; r[2,2]=(m00*m11-m01*m10)*inv_det
    return r


def mat4_to_f32_col_major(m: np.ndarray) -> np.ndarray:
    r=np.empty(16, dtype=np.float32)
    for i in range(4):
        i4=i*4
        r[i4]=np.float32(m[0,i]); r[i4+1]=np.float32(m[1,i])
        r[i4+2]=np.float32(m[2,i]); r[i4+3]=np.float32(m[3,i])
    return r


def mat4_mul_vec3(m: np.ndarray, vx: float, vy: float, vz: float):
    return (
        m[0,0]*vx+m[1,0]*vy+m[2,0]*vz+m[3,0],
        m[0,1]*vx+m[1,1]*vy+m[2,1]*vz+m[3,1],
        m[0,2]*vx+m[1,2]*vy+m[2,2]*vz+m[3,2]
    )


def mat4_look_at(eye_x: float, eye_y: float, eye_z: float,
                 center_x: float, center_y: float, center_z: float,
                 up_x: float, up_y: float, up_z: float) -> np.ndarray:
    fx=center_x-eye_x; fy=center_y-eye_y; fz=center_z-eye_z
    flen=(fx*fx+fy*fy+fz*fz)**0.5
    if flen > 1e-10:
        inv=1.0/flen; fx*=inv; fy*=inv; fz*=inv
    rx=up_y*fz-up_z*fy; ry=up_z*fx-up_x*fz; rz=up_x*fy-up_y*fx
    rlen=(rx*rx+ry*ry+rz*rz)**0.5
    if rlen > 1e-10:
        inv=1.0/rlen; rx*=inv; ry*=inv; rz*=inv
    ux=fy*rz-fz*ry; uy=fz*rx-fx*rz; uz=fx*ry-fy*rx
    m=np.eye(4, dtype=FLOAT_TYPE)
    m[0,0]=rx; m[1,0]=ry; m[2,0]=rz
    m[0,1]=ux; m[1,1]=uy; m[2,1]=uz
    m[0,2]=-fx; m[1,2]=-fy; m[2,2]=-fz
    m[3,0]=-(rx*eye_x+ry*eye_y+rz*eye_z)
    m[3,1]=-(ux*eye_x+uy*eye_y+uz*eye_z)
    m[3,2]=fx*eye_x+fy*eye_y+fz*eye_z
    return m


def mat4_perspective(fov_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f=1.0/math.tan(fov_rad*0.5)
    nf=1.0/(near-far)
    m=np.zeros((4,4), dtype=FLOAT_TYPE)
    m[0,0]=f/aspect; m[1,1]=f
    m[2,2]=(far+near)*nf; m[2,3]=-1.0
    m[3,2]=2.0*far*near*nf
    return m


def vec3_normalize(x: float, y: float, z: float):
    n=(x*x+y*y+z*z)**0.5
    if n > 1e-10:
        inv_n=1.0/n
        return x*inv_n,y*inv_n,z*inv_n
    return 0.0,0.0,0.0


def vec3_sub(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ax-bx,ay-by,az-bz


def vec3_add(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ax+bx,ay+by,az+bz


def vec3_scale(x: float, y: float, z: float, s: float):
    return x*s,y*s,z*s


def vec3_dot(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ax*bx+ay*by+az*bz


def vec3_cross(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx


def ray_triangle_intersect(ox, oy, oz, dx, dy, dz,
                           v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    e1x = v1x - v0x; e1y = v1y - v0y; e1z = v1z - v0z
    e2x = v2x - v0x; e2y = v2y - v0y; e2z = v2z - v0z
    px = dy * e2z - dz * e2y; py = dz * e2x - dx * e2z; pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-12:
        return -1.0
    inv_det = 1.0 / det
    tx = ox - v0x; ty = oy - v0y; tz = oz - v0z
    u = (tx * px + ty * py + tz * pz) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0
    qx = ty * e1z - tz * e1y; qy = tz * e1x - tx * e1z; qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return -1.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    return t if t > 0.0 else -1.0


def ray_mesh_intersect(ox, oy, oz, dx, dy, dz,
                       verts, indices):
    best_t = -1.0
    for i in range(0, len(indices), 3):
        i0 = indices[i]; i1 = indices[i+1]; i2 = indices[i+2]
        v0x = verts[i0*3]; v0y = verts[i0*3+1]; v0z = verts[i0*3+2]
        v1x = verts[i1*3]; v1y = verts[i1*3+1]; v1z = verts[i1*3+2]
        v2x = verts[i2*3]; v2y = verts[i2*3+1]; v2z = verts[i2*3+2]
        t = ray_triangle_intersect(ox, oy, oz, dx, dy, dz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)
        if t > 0 and (best_t < 0 or t < best_t):
            best_t = t
    return best_t


def ray_aabb_intersect(ox: float, oy: float, oz: float,
                       dx: float, dy: float, dz: float,
                       bmin_x: float, bmin_y: float, bmin_z: float,
                       bmax_x: float, bmax_y: float, bmax_z: float) -> float:
    tmin=-1e30; tmax=1e30
    if abs(dx) > 1e-30:
        t1=(bmin_x-ox)/dx; t2=(bmax_x-ox)/dx
        if t1>t2: t1,t2=t2,t1
        if t1>tmin: tmin=t1
        if t2<tmax: tmax=t2
    elif ox<bmin_x or ox>bmax_x:
        return -1.0
    if abs(dy) > 1e-30:
        t1=(bmin_y-oy)/dy; t2=(bmax_y-oy)/dy
        if t1>t2: t1,t2=t2,t1
        if t1>tmin: tmin=t1
        if t2<tmax: tmax=t2
    elif oy<bmin_y or oy>bmax_y:
        return -1.0
    if abs(dz) > 1e-30:
        t1=(bmin_z-oz)/dz; t2=(bmax_z-oz)/dz
        if t1>t2: t1,t2=t2,t1
        if t1>tmin: tmin=t1
        if t2<tmax: tmax=t2
    elif oz<bmin_z or oz>bmax_z:
        return -1.0
    if tmin>tmax: return -1.0
    return tmin if tmin>0.0 else (tmax if tmax>0.0 else -1.0)


def mat4_normal_matrix(model: np.ndarray) -> np.ndarray:
    m = model[:3, :3]
    try:
        inv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)
    return inv.T.astype(np.float32)


def batch_matrices_to_f32(matrices: np.ndarray) -> np.ndarray:
    n = matrices.shape[0]
    if n == 0:
        return np.zeros((0, 16), dtype=np.float32)
    return matrices.reshape(n, 16).astype(np.float32)