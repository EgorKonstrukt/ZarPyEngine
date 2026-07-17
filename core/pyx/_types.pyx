# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from libc.math cimport sqrt, sin, cos, fabs

ctypedef double DTYPE_t

cdef class Vec3d:
    cdef public DTYPE_t x, y, z

    def __cinit__(self, DTYPE_t x=0.0, DTYPE_t y=0.0, DTYPE_t z=0.0):
        self.x = x
        self.y = y
        self.z = z

    cdef inline Vec3d _make(self, DTYPE_t x, DTYPE_t y, DTYPE_t z):
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = x; v.y = y; v.z = z
        return v

    def __add__(Vec3d self, Vec3d o):
        return self._make(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(Vec3d self, Vec3d o):
        return self._make(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        cdef DTYPE_t _s = <DTYPE_t>s
        return self._make(self.x * _s, self.y * _s, self.z * _s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def __truediv__(self, s):
        cdef DTYPE_t _s = <DTYPE_t>s
        return self._make(self.x / _s, self.y / _s, self.z / _s)

    def __neg__(self):
        return self._make(-self.x, -self.y, -self.z)

    def __repr__(self):
        return f"Vec3d({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, o):
        if not isinstance(o, Vec3d):
            return False
        cdef Vec3d v = <Vec3d>o
        return fabs(self.x - v.x) < 1e-8 and fabs(self.y - v.y) < 1e-8 and fabs(self.z - v.z) < 1e-8

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __getitem__(self, int i):
        if i == 0: return self.x
        if i == 1: return self.y
        if i == 2: return self.z
        raise IndexError(i)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    cdef inline DTYPE_t dot(Vec3d self, Vec3d o):
        return self.x*o.x + self.y*o.y + self.z*o.z

    cdef inline Vec3d cross(Vec3d self, Vec3d o):
        return self._make(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x,
        )

    cdef inline DTYPE_t length(Vec3d self):
        return sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    cdef inline DTYPE_t length_sq(Vec3d self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    cdef inline Vec3d normalized(Vec3d self):
        cdef DTYPE_t l = sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        if l > 1e-10:
            l = 1.0 / l
            return self._make(self.x*l, self.y*l, self.z*l)
        return self._make(0.0, 0.0, 0.0)

    cdef inline Vec3d lerp(Vec3d self, Vec3d o, DTYPE_t t):
        return self._make(
            self.x + (o.x - self.x)*t,
            self.y + (o.y - self.y)*t,
            self.z + (o.z - self.z)*t,
        )

    def dot_py(self, Vec3d o):
        return self.dot(o)

    def cross_py(self, Vec3d o):
        return self.cross(o)

    def length_py(self):
        return self.length()

    def length_sq_py(self):
        return self.length_sq()

    def normalized_py(self):
        return self.normalized()

    def lerp_py(self, Vec3d o, DTYPE_t t):
        return self.lerp(o, t)

    def to_list(self):
        return [self.x, self.y, self.z]

    @staticmethod
    def zero():
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = 0.0; v.y = 0.0; v.z = 0.0
        return v

    @staticmethod
    def one():
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = 1.0; v.y = 1.0; v.z = 1.0
        return v

    @staticmethod
    def up():
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = 0.0; v.y = 1.0; v.z = 0.0
        return v

    @staticmethod
    def forward():
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = 0.0; v.y = 0.0; v.z = -1.0
        return v

    @staticmethod
    def right():
        cdef Vec3d v = Vec3d.__new__(Vec3d)
        v.x = 1.0; v.y = 0.0; v.z = 0.0
        return v


cdef class Mat4d:
    cdef DTYPE_t _d[16]

    def __cinit__(self):
        self._d[0]=1; self._d[5]=1; self._d[10]=1; self._d[15]=1

    cdef inline void _mul(Mat4d self, Mat4d r, Mat4d o) noexcept nogil:
        cdef DTYPE_t* a = &self._d[0]
        cdef DTYPE_t* b = &r._d[0]
        cdef DTYPE_t* c = &o._d[0]
        cdef int i
        cdef DTYPE_t a0, a1, a2, a3
        for i in range(4):
            a0 = a[i]; a1 = a[4+i]; a2 = a[8+i]; a3 = a[12+i]
            c[i]    = a0*b[0]  + a1*b[1]  + a2*b[2]  + a3*b[3]
            c[4+i]  = a0*b[4]  + a1*b[5]  + a2*b[6]  + a3*b[7]
            c[8+i]  = a0*b[8]  + a1*b[9]  + a2*b[10] + a3*b[11]
            c[12+i] = a0*b[12] + a1*b[13] + a2*b[14] + a3*b[15]

    cdef inline void _inv(Mat4d self, Mat4d o) noexcept nogil:
        cdef DTYPE_t* m = &self._d[0]
        cdef DTYPE_t* o_ptr = &o._d[0]
        cdef DTYPE_t m00=m[0], m01=m[1], m02=m[2], m03=m[3]
        cdef DTYPE_t m10=m[4], m11=m[5], m12=m[6], m13=m[7]
        cdef DTYPE_t m20=m[8], m21=m[9], m22=m[10], m23=m[11]
        cdef DTYPE_t m30=m[12], m31=m[13], m32=m[14], m33=m[15]
        cdef DTYPE_t t00 = m11*m22*m33 - m11*m23*m32 - m12*m21*m33 + m12*m23*m31 + m13*m21*m32 - m13*m22*m31
        cdef DTYPE_t t10 = -m10*m22*m33 + m10*m23*m32 + m12*m20*m33 - m12*m23*m30 - m13*m20*m32 + m13*m22*m30
        cdef DTYPE_t t20 = m10*m21*m33 - m10*m23*m31 - m11*m20*m33 + m11*m23*m30 + m13*m20*m31 - m13*m21*m30
        cdef DTYPE_t t30 = -m10*m21*m32 + m10*m22*m31 + m11*m20*m32 - m11*m22*m30 - m12*m20*m31 + m12*m21*m30
        cdef DTYPE_t det = m00*t00 + m01*t10 + m02*t20 + m03*t30
        if fabs(det) < 1e-15:
            o_ptr[0]=1; o_ptr[5]=1; o_ptr[10]=1; o_ptr[15]=1
            o_ptr[1]=0; o_ptr[2]=0; o_ptr[3]=0; o_ptr[4]=0
            o_ptr[6]=0; o_ptr[7]=0; o_ptr[8]=0; o_ptr[9]=0
            o_ptr[11]=0; o_ptr[12]=0; o_ptr[13]=0; o_ptr[14]=0
            return
        cdef DTYPE_t inv_det = 1.0 / det
        o_ptr[0]  = t00 * inv_det
        o_ptr[4]  = t10 * inv_det
        o_ptr[8]  = t20 * inv_det
        o_ptr[12] = t30 * inv_det
        o_ptr[1]  = (-m01*m22*m33 + m01*m23*m32 + m02*m21*m33 - m02*m23*m31 - m03*m21*m32 + m03*m22*m31) * inv_det
        o_ptr[5]  = ( m00*m22*m33 - m00*m23*m32 - m02*m20*m33 + m02*m23*m30 + m03*m20*m32 - m03*m22*m30) * inv_det
        o_ptr[9]  = (-m00*m21*m33 + m00*m23*m31 + m01*m20*m33 - m01*m23*m30 - m03*m20*m31 + m03*m21*m30) * inv_det
        o_ptr[13] = ( m00*m21*m32 - m00*m22*m31 - m01*m20*m32 + m01*m22*m30 + m02*m20*m31 - m02*m21*m30) * inv_det
        o_ptr[2]  = ( m01*m12*m33 - m01*m13*m32 - m02*m11*m33 + m02*m13*m31 + m03*m11*m32 - m03*m12*m31) * inv_det
        o_ptr[6]  = (-m00*m12*m33 + m00*m13*m32 + m02*m10*m33 - m02*m13*m30 - m03*m10*m32 + m03*m12*m30) * inv_det
        o_ptr[10] = ( m00*m11*m33 - m00*m13*m31 - m01*m10*m33 + m01*m13*m30 + m03*m10*m31 - m03*m11*m30) * inv_det
        o_ptr[14] = (-m00*m11*m32 + m00*m12*m31 + m01*m10*m32 - m01*m12*m30 - m02*m10*m31 + m02*m11*m30) * inv_det
        o_ptr[3]  = (-m01*m12*m23 + m01*m13*m22 + m02*m11*m23 - m02*m13*m21 - m03*m11*m22 + m03*m12*m21) * inv_det
        o_ptr[7]  = ( m00*m12*m23 - m00*m13*m22 - m02*m10*m23 + m02*m13*m20 + m03*m10*m22 - m03*m12*m20) * inv_det
        o_ptr[11] = (-m00*m11*m23 + m00*m13*m21 + m01*m10*m23 - m01*m13*m20 - m03*m10*m21 + m03*m11*m20) * inv_det
        o_ptr[15] = ( m00*m11*m22 - m00*m12*m21 - m01*m10*m22 + m01*m12*m20 + m02*m10*m21 - m02*m11*m20) * inv_det

    cdef inline void _mul_vec3(Mat4d self,
                                DTYPE_t x, DTYPE_t y, DTYPE_t z,
                                DTYPE_t* rx, DTYPE_t* ry, DTYPE_t* rz) noexcept nogil:
        cdef DTYPE_t* m = &self._d[0]
        rx[0] = m[0]*x + m[4]*y + m[8]*z  + m[12]
        ry[0] = m[1]*x + m[5]*y + m[9]*z  + m[13]
        rz[0] = m[2]*x + m[6]*y + m[10]*z + m[14]

    def __mul__(Mat4d self, Mat4d r):
        cdef Mat4d o = Mat4d.__new__(Mat4d)
        self._mul(r, o)
        return o

    def inv(Mat4d self):
        cdef Mat4d o = Mat4d.__new__(Mat4d)
        self._inv(o)
        return o

    def mul_vec3(Mat4d self, Vec3d v):
        cdef DTYPE_t rx, ry, rz
        self._mul_vec3(v.x, v.y, v.z, &rx, &ry, &rz)
        cdef Vec3d out = Vec3d.__new__(Vec3d)
        out.x = rx; out.y = ry; out.z = rz
        return out

    def get_translation(Mat4d self):
        cdef Vec3d out = Vec3d.__new__(Vec3d)
        out.x = self._d[12]; out.y = self._d[13]; out.z = self._d[14]
        return out

    def set_translation(Mat4d self, Vec3d v):
        self._d[12] = v.x; self._d[13] = v.y; self._d[14] = v.z

    def to_list(Mat4d self):
        return [self._d[i] for i in range(16)]

    @staticmethod
    def identity():
        cdef Mat4d m = Mat4d.__new__(Mat4d)
        m._d[0]=1; m._d[5]=1; m._d[10]=1; m._d[15]=1
        return m

    @staticmethod
    def translation(Vec3d v):
        cdef Mat4d m = Mat4d.__new__(Mat4d)
        m._d[0]=1; m._d[5]=1; m._d[10]=1; m._d[15]=1
        m._d[12] = v.x; m._d[13] = v.y; m._d[14] = v.z
        return m

    @staticmethod
    def scale(Vec3d v):
        cdef Mat4d m = Mat4d.__new__(Mat4d)
        m._d[0] = v.x; m._d[5] = v.y; m._d[10] = v.z; m._d[15] = 1.0
        return m

    @staticmethod
    def perspective(DTYPE_t fov_deg, DTYPE_t aspect, DTYPE_t near, DTYPE_t far):
        cdef DTYPE_t f = 1.0 / cos(fov_deg * 3.14159265358979323846 / 360.0)
        cdef DTYPE_t nf = 1.0 / (near - far)
        cdef Mat4d m = Mat4d.__new__(Mat4d)
        m._d[0] = f / aspect; m._d[5] = f
        m._d[10] = (far + near) * nf; m._d[11] = -1.0
        m._d[14] = 2.0 * far * near * nf
        m._d[15] = 0.0
        return m

    @staticmethod
    def look_at(Vec3d eye, Vec3d center, Vec3d up):
        cdef DTYPE_t fx = center.x - eye.x, fy = center.y - eye.y, fz = center.z - eye.z
        cdef DTYPE_t fl = sqrt(fx*fx + fy*fy + fz*fz)
        if fl > 1e-10: fx /= fl; fy /= fl; fz /= fl
        cdef DTYPE_t rx = up.y*fz - up.z*fy, ry = up.z*fx - up.x*fz, rz = up.x*fy - up.y*fx
        cdef DTYPE_t rl = sqrt(rx*rx + ry*ry + rz*rz)
        if rl > 1e-10: rx /= rl; ry /= rl; rz /= rl
        cdef DTYPE_t ux = fy*rz - fz*ry, uy = fz*rx - fx*rz, uz = fx*ry - fy*rx
        cdef Mat4d m = Mat4d.__new__(Mat4d)
        m._d[0]=rx;  m._d[1]=ry;  m._d[2]=rz
        m._d[4]=ux;  m._d[5]=uy;  m._d[6]=uz
        m._d[8]=-fx; m._d[9]=-fy; m._d[10]=-fz
        m._d[12]=-(rx*eye.x + ry*eye.y + rz*eye.z)
        m._d[13]=-(ux*eye.x + uy*eye.y + uz*eye.z)
        m._d[14]=fx*eye.x + fy*eye.y + fz*eye.z
        m._d[15]=1.0
        return m
