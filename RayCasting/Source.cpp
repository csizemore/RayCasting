
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		 
#include <GL/freeglut.h>	
#include <vector>	

#endif

const unsigned int windowWidth = 500, windowHeight = 500;

int majorVersion = 3, minorVersion = 0;

class vec2
{
public:
	float x;
	float y;

	vec2()
	{
		x = 0.0f;
		y = 0.0f;
	}


	vec2(float x, float y) :x(x), y(y) {}

	vec2 operator-() const
	{
		return vec2(-x, -y);
	}


	vec2 operator+(const vec2& operand) const
	{
		return vec2(x + operand.x, y + operand.y);
	}

	vec2 operator-(const vec2& operand) const
	{
		return vec2(x - operand.x, y - operand.y);
	}

	vec2 operator*(const vec2& operand) const
	{
		return vec2(x * operand.x, y * operand.y);
	}

	vec2 operator*(float operand) const
	{
		return vec2(x * operand, y * operand);
	}

	void operator-=(const vec2& a)
	{
		x -= a.x;
		y -= a.y;
	}

	void operator+=(const vec2& a)
	{
		x += a.x;
		y += a.y;
	}

	void operator*=(const vec2& a)
	{
		x *= a.x;
		y *= a.y;
	}

	void operator*=(float a)
	{
		x *= a;
		y *= a;
	}

	float norm()
	{
		return sqrtf(x*x + y*y);
	}

	float norm2()
	{
		return x*x + y*y;
	}

	vec2 normalize()
	{
		float oneOverLength = 1.0f / norm();
		x *= oneOverLength;
		y *= oneOverLength;
		return *this;
	}

	static vec2 random()
	{
		return vec2(
			((float)rand() / RAND_MAX) * 2 - 1,
			((float)rand() / RAND_MAX) * 2 - 1);
	}
};
class vec3
{
public:
	float x;
	float y;
	float z;

	vec3()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	static vec3 random()
	{
		return vec3(
			((float)rand() / RAND_MAX),
			((float)rand() / RAND_MAX),
			((float)rand() / RAND_MAX));
	}

	vec3(float x, float y, float z) :x(x), y(y), z(z) {}

	vec3 operator-() const
	{
		return vec3(-x, -y, -z);
	}

	vec3 operator+(const vec3& operand) const
	{
		return vec3(x + operand.x, y + operand.y, z + operand.z);
	}

	vec3 operator-(const vec3& operand) const
	{
		return vec3(x - operand.x, y - operand.y, z - operand.z);
	}

	vec3 operator*(const vec3& operand) const
	{
		return vec3(x * operand.x, y * operand.y, z * operand.z);
	}

	vec3 operator*(float operand) const
	{
		return vec3(x * operand, y * operand, z * operand);
	}

	void operator-=(const vec3& a)
	{
		x -= a.x;
		y -= a.y;
		z -= a.z;
	}

	void operator+=(const vec3& a)
	{
		x += a.x;
		y += a.y;
		z += a.z;
	}

	void operator*=(const vec3& a)
	{
		x *= a.x;
		y *= a.y;
		z *= a.z;
	}

	void operator*=(float a)
	{
		x *= a;
		y *= a;
		z *= a;
	}

	vec3 operator/(float operand) const
	{
		if (operand == 0)
			return vec3(0, 0, 0);
		else
			return vec3(x / operand, y / operand, z / operand);
	}

	vec3 operator/(vec3 operand) const
	{
		if (operand.x == 0 || operand.y == 0 || operand.z == 0)
			return vec3(0, 0, 0);
		else
			return vec3(x / operand.x, y / operand.y, z / operand.z);
	}

	float norm() const
	{
		return sqrtf(x*x + y*y + z*z);
	}

	float norm2() const
	{
		return x*x + y*y + z*z;
	}

	vec3 normalize()
	{
		float oneOverLength = 1.0f / norm();
		x *= oneOverLength;
		y *= oneOverLength;
		z *= oneOverLength;
		return *this;
	}

	vec3 cross(const vec3& operand) const
	{
		return vec3(
			y * operand.z - z * operand.y,
			z * operand.x - x * operand.z,
			x * operand.y - y * operand.x);
	}

	float dot(const vec3& operand) const
	{
		return x * operand.x + y * operand.y + z * operand.z;
	}
};
class vec4
{
public:
	union {
		struct {
			float x;
			float y;
			float z;
			float w;
		};

		float v[4];
	};

	vec4() :x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

	vec4(float f) :x(f), y(f), z(f), w(f) {}

	vec4(vec3 f3) :x(f3.x), y(f3.y), z(f3.z), w(1.0f) {}

	vec4(float x, float y, float z, float w) :x(x), y(y), z(z), w(w) {}

	vec4& operator+=(const vec4& o)
	{
		x += o.x;
		y += o.y;
		z += o.z;
		w += o.w;
		return *this;
	}

	vec4& operator-=(const vec4& o)
	{
		x -= o.x;
		y -= o.y;
		z -= o.z;
		w -= o.w;
		return *this;
	}

	vec4& operator*=(const vec4& o)
	{
		x *= o.x;
		y *= o.y;
		z *= o.z;
		w *= o.w;
		return *this;
	}

	vec4& operator/=(const vec4& o)
	{
		x /= o.x;
		y /= o.y;
		z /= o.z;
		w /= o.w;
		return *this;
	}

	vec4& operator%=(const vec4& o)
	{
		x = fmodf(x, o.x);
		y = fmodf(y, o.y);
		z = fmodf(z, o.z);
		w = fmodf(w, o.w);
		return *this;
	}

	vec4 operator+(const vec4& o) const
	{
		return vec4(x + o.x, y + o.y, z + o.z, w + o.w);
	}

	vec4 operator-(const vec4& o) const
	{
		return vec4(x - o.x, y - o.y, z - o.z, w - o.w);
	}

	vec4 operator*(const vec4& o) const
	{
		return vec4(x * o.x, y * o.y, z * o.z, w * o.w);
	}

	vec4 operator/(const vec4& o) const
	{
		return vec4(x / o.x, y / o.y, z / o.z, w / o.w);
	}

	vec4 operator%(const vec4& o) const
	{
		return vec4(fmodf(x, o.x), fmodf(y, o.y), fmodf(z, o.z), fmodf(w, o.w));
	}


	vec4 operator+() const
	{
		return vec4(+x, +y, +z, +w);
	}

	vec4 operator-() const
	{
		return vec4(-x, -y, -z, -w);
	}

	vec4 operator!() const
	{
		return vec4(-x, -y, -z, +w);
	}

	float distance(const vec4& o) const
	{
		return (*this - o).norm();
	}

	float dot(const vec4& o) const
	{
		return x * o.x + y * o.y + z * o.z + w * o.w;
	}

	float norm() const
	{
		return sqrtf(this->dot(*this));
	}

	float norm2() const
	{
		return this->dot(*this);
	}

	vec4 normalize() const
	{
		return *this / norm();
	}

};
class mat4x4
{
public:
	union
	{
		struct
		{
			float        _00, _01, _02, _03;
			float        _10, _11, _12, _13;
			float        _20, _21, _22, _23;
			float        _30, _31, _32, _33;
		};
		float m[4][4];
		float l[16];
	};

	mat4x4() :
		_00(1.0f), _01(0.0f), _02(0.0f), _03(0.0f),
		_10(0.0f), _11(1.0f), _12(0.0f), _13(0.0f),
		_20(0.0f), _21(0.0f), _22(1.0f), _23(0.0f),
		_30(0.0f), _31(0.0f), _32(0.0f), _33(1.0f)
	{
	}

	mat4x4(
		float _00, float _01, float _02, float _03,
		float _10, float _11, float _12, float _13,
		float _20, float _21, float _22, float _23,
		float _30, float _31, float _32, float _33) :
		_00(_00), _01(_01), _02(_02), _03(_03),
		_10(_10), _11(_11), _12(_12), _13(_13),
		_20(_20), _21(_21), _22(_22), _23(_23),
		_30(_30), _31(_31), _32(_32), _33(_33)
	{
	}

	mat4x4 elementwiseProduct(const mat4x4& o) const
	{
		mat4x4 r;
		for (int i = 0; i<16; i++)
			r.l[i] = l[i] * o.l[i];
		return r;
	}

	mat4x4 operator+(const mat4x4& o) const
	{
		mat4x4 r;
		for (int i = 0; i<16; i++)
			r.l[i] = l[i] + o.l[i];
		return r;
	}

	mat4x4 operator-(const mat4x4& o) const
	{
		mat4x4 r;
		for (int i = 0; i<16; i++)
			r.l[i] = l[i] - o.l[i];
		return r;
	}

	mat4x4& assignElementwiseProduct(const mat4x4& o)
	{
		for (int i = 0; i<16; i++)
			l[i] *= o.l[i];
		return *this;
	}

	mat4x4& operator*=(float s)
	{
		for (int i = 0; i<16; i++)
			l[i] *= s;
		return *this;
	}

	mat4x4& operator/=(float s)
	{
		float is = 1 / s;
		for (int i = 0; i<16; i++)
			l[i] *= is;
		return *this;
	}

	mat4x4& operator+=(const mat4x4& o)
	{
		for (int i = 0; i<16; i++)
			l[i] += o.l[i];
		return *this;
	}

	mat4x4& operator-=(const mat4x4& o)
	{
		for (int i = 0; i<16; i++)
			l[i] -= o.l[i];
		return *this;
	}

	mat4x4 mul(const mat4x4& o) const
	{
		mat4x4 product;

		for (int r = 0; r<4; r++)
			for (int c = 0; c<4; c++)
				product.m[r][c] =
				m[r][0] * o.m[0][c] +
				m[r][1] * o.m[1][c] +
				m[r][2] * o.m[2][c] +
				m[r][3] * o.m[3][c];

		return product;
	}

	mat4x4 operator<<(const mat4x4& o) const
	{
		return mul(o);
	}

	mat4x4& operator <<=(const mat4x4& o)
	{
		*this = *this << o;
		return *this;
	}

	mat4x4 operator*(const mat4x4& o) const
	{
		return mul(o);
	}

	mat4x4& operator*=(const mat4x4& o)
	{
		*this = *this * o;
		return *this;
	}

	vec4 mul(const vec4& v) const
	{
		return vec4(v.dot(*(vec4*)m[0]), v.dot(*(vec4*)m[1]), v.dot(*(vec4*)m[2]), v.dot(*(vec4*)m[3]));
	}

	vec4 transform(const vec4& v) const
	{
		return vec4(
			_00 * v.x + _10 * v.y + _20 * v.z + _30 * v.w,
			_01 * v.x + _11 * v.y + _21 * v.z + _31 * v.w,
			_02 * v.x + _12 * v.y + _22 * v.z + _32 * v.w,
			_03 * v.x + _13 * v.y + _23 * v.z + _33 * v.w
		);
	}

	vec4 operator*(const vec4& v) const
	{
		return mul(v);
	}

	mat4x4 operator*(float s) const
	{
		mat4x4 r;
		for (int i = 0; i<16; i++)
			r.l[i] = l[i] * s;
		return r;
	}

	static const mat4x4 identity()
	{
		return mat4x4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	static mat4x4 scaling(const vec3& factors)
	{
		mat4x4 s = identity();
		s._00 = factors.x;
		s._11 = factors.y;
		s._22 = factors.z;

		return s;
	}

	static mat4x4 translation(const vec3& offset)
	{
		mat4x4 t = identity();
		t._30 = offset.x;
		t._31 = offset.y;
		t._32 = offset.z;
		return t;
	}

	static mat4x4 rotation(const vec3& axis, float angle)
	{
		mat4x4 r = identity();

		float s = sin(angle);
		float c = cos(angle);
		float t = 1 - c;

		float axisLength = axis.norm();
		if (axisLength == 0.0f)
			return identity();
		vec3 ax = axis * (1.0f / axisLength);

		float& x = ax.x;
		float& y = ax.y;
		float& z = ax.z;

		r._00 = t*x*x + c;
		r._01 = t*y*x + s*z;
		r._02 = t*z*x - s*y;

		r._10 = t*x*y - s*z;
		r._11 = t*y*y + c;
		r._12 = t*z*y + s*x;

		r._20 = t*x*z + s*y;
		r._21 = t*y*z - s*x;
		r._22 = t*z*z + c;

		return r;
	}

	mat4x4 transpose() const
	{
		return mat4x4(
			_00, _10, _20, _30,
			_01, _11, _21, _31,
			_02, _12, _22, _32,
			_03, _13, _23, _33);
	}

	mat4x4 _invert() const
	{
		float det;
		float d10, d20, d21, d31, d32, d03;
		mat4x4 inv;

		/* Inverse = adjoint / det. (See linear algebra texts.)*/

		/* pre-compute 2x2 dets for last two rows when computing */
		/* cofactors of first two rows. */
		d10 = (_02*_13 - _03*_12);
		d20 = (_02*_23 - _03*_22);
		d21 = (_12*_23 - _13*_22);
		d31 = (_12*_33 - _13*_32);
		d32 = (_22*_33 - _23*_32);
		d03 = (_32*_03 - _33*_02);

		inv.l[0] = (_11 * d32 - _21 * d31 + _31 * d21);
		inv.l[1] = -(_01 * d32 + _21 * d03 + _31 * d20);
		inv.l[2] = (_01 * d31 + _11 * d03 + _31 * d10);
		inv.l[3] = -(_01 * d21 - _11 * d20 + _21 * d10);

		/* Compute determinant as early as possible using these cofactors. */
		det = _00 * inv.l[0] + _10 * inv.l[1] + _20 * inv.l[2] + _30 * inv.l[3];

		/* Run singularity test. */
		if (det == 0.0) {
			return identity();
		}
		else
		{
			float invDet = 1.0 / det;
			/* Compute rest of inverse. */
			inv.l[0] *= invDet;
			inv.l[1] *= invDet;
			inv.l[2] *= invDet;
			inv.l[3] *= invDet;

			inv.l[4] = -(_10 * d32 - _20 * d31 + _30 * d21) * invDet;
			inv.l[5] = (_00 * d32 + _20 * d03 + _30 * d20) * invDet;
			inv.l[6] = -(_00 * d31 + _10 * d03 + _30 * d10) * invDet;
			inv.l[7] = (_00 * d21 - _10 * d20 + _20 * d10) * invDet;

			/* Pre-compute 2x2 dets for first two rows when computing */
			/* cofactors of last two rows. */
			d10 = _00*_11 - _01*_10;
			d20 = _00*_21 - _01*_20;
			d21 = _10*_21 - _11*_20;
			d31 = _10*_31 - _11*_30;
			d32 = _20*_31 - _21*_30;
			d03 = _30*_01 - _31*_00;

			inv.l[8] = (_13 * d32 - _23 * d31 + _33 * d21) * invDet;
			inv.l[9] = -(_03 * d32 + _23 * d03 + _33 * d20) * invDet;
			inv.l[10] = (_03 * d31 + _13 * d03 + _33 * d10) * invDet;
			inv.l[11] = -(_03 * d21 - _13 * d20 + _23 * d10) * invDet;
			inv.l[12] = -(_12 * d32 - _22 * d31 + _32 * d21) * invDet;
			inv.l[13] = (_02 * d32 + _22 * d03 + _32 * d20) * invDet;
			inv.l[14] = -(_02 * d31 + _12 * d03 + _32 * d10) * invDet;
			inv.l[15] = (_02 * d21 - _12 * d20 + _22 * d10) * invDet;

			return inv;
		}
	}

	mat4x4 invert() const
	{
		register float det;

		if (_03 != 0.0f || _13 != 0.0f || _23 != 0.0f || _33 != 1.0f)
		{
			return _invert();
		}

		mat4x4 inv;

		/* Inverse = adjoint / det. */
		inv.l[0] = _11 * _22 - _21 * _12;
		inv.l[1] = _21 * _02 - _01 * _22;
		inv.l[2] = _01 * _12 - _11 * _02;

		/* Compute determinant as early as possible using these cofactors. */
		det = _00 * inv.l[0] + _10 * inv.l[1] + _20 * inv.l[2];

		/* Run singularity test. */
		if (det == 0.0f)
		{
			/* printf("invert_mat4x4: Warning: Singular mat4x4.\n"); */
			return identity();
		}
		else
		{
			float d10, d20, d21, d31, d32, d03;
			register float im00, im10, im20, im30;

			det = 1.0f / det;

			/* Compute rest of inverse. */
			inv.l[0] *= det;
			inv.l[1] *= det;
			inv.l[2] *= det;
			inv.l[3] = 0.0f;

			im00 = _00 * det;
			im10 = _10 * det;
			im20 = _20 * det;
			im30 = _30 * det;
			inv.l[4] = im20 * _12 - im10 * _22;
			inv.l[5] = im00 * _22 - im20 * _02;
			inv.l[6] = im10 * _02 - im00 * _12;
			inv.l[7] = 0.0f;

			/* Pre-compute 2x2 dets for first two rows when computing */
			/* cofactors of last two rows. */
			d10 = im00 * _11 - _01 * im10;
			d20 = im00 * _21 - _01 * im20;
			d21 = im10 * _21 - _11 * im20;
			d31 = im10 * _31 - _11 * im30;
			d32 = im20 * _31 - _21 * im30;
			d03 = im30 * _01 - _31 * im00;

			inv.l[8] = d21;
			inv.l[9] = -d20;
			inv.l[10] = d10;
			inv.l[11] = 0.0f;

			inv.l[12] = -(_12 * d32 - _22 * d31 + _32 * d21);
			inv.l[13] = (_02 * d32 + _22 * d03 + _32 * d20);
			inv.l[14] = -(_02 * d31 + _12 * d03 + _32 * d10);
			inv.l[15] = 1.0f;

			return inv;
		}
	}

};
inline vec4 operator*(const vec4& v, const mat4x4& m)
{
	return m.transform(v);
}
inline const vec4& operator*=(vec4& v, const mat4x4& m)
{
	v = m.transform(v);
	return v;
}

// image to be computed by ray tracing
vec3 image[windowWidth * windowHeight];

// simple material class, with object color, and headlight shading
class Material
{

public:
	Material()
	{}

	virtual vec3 shade(vec3 powerDensity, vec3 lightDirection, 
		vec3 hitNormal, vec3 viewDirection, vec3 position) = 0;

	float snoise(vec3 r) {
		unsigned int x = 0x0625DF73;
		unsigned int y = 0xD1B84B45;
		unsigned int z = 0x152AD8D0;
		float f = 0;
		for (int i = 0; i<32; i++) {
			vec3 s(x / (float)0xffffffff,
				y / (float)0xffffffff,
				z / (float)0xffffffff);
			f += sin(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f / 64.0 + 0.5;
	}

};

class DiffuseMaterial :
	public Material
{

	vec3 color;
public:
	DiffuseMaterial(vec3 colora) 
	{
		color = colora;
	}

	vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		return powerDensity * color * (hitNormal.dot(lightDirection));
	}

};

class Emissive : public Material
{
	vec3 color;
public:
	Emissive(vec3 colora)
	{
		color = colora;
	}

	vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		return powerDensity * color * (hitNormal.dot(lightDirection));
	}

};

class EmissiveTwo : public Material
{
	vec3 color;
public:
	EmissiveTwo(vec3 colora)
	{
		color = colora;
	}

	vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		return color;
	}

};

class Wood :
	public Material
{
	float scale;
	float turbulence;
	float period;
	float sharpness;
public:
	Wood()
	{
		scale = 16;
		turbulence = 500;
		period = 8;
		sharpness = 10;
	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		//return normal;
		float w = position.x * period + pow(snoise(position * scale), sharpness)
			*turbulence + 10000.0;
		w -= int(w);
		return (vec3(1, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1 - w)) * normal.dot(viewDir);
	}

	vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		return powerDensity * getColor(position, hitNormal, viewDirection) 
			* (hitNormal.dot(lightDirection));
	}
};

class ReflectiveWood : public Material
{
	vec3 r0;

	float scale;
	float turbulence;
	float period;
	float sharpness;
public:
	ReflectiveWood(vec3  refractiveIndex, vec3  extinctionCoefficient)
	{
		scale = 16;
		turbulence = 500;
		period = 8;
		sharpness = 10;

		vec3 rim1 = refractiveIndex - vec3(1, 1, 1);
		vec3 rip1 = refractiveIndex + vec3(1, 1, 1);
		vec3 k2 = extinctionCoefficient * extinctionCoefficient;
		r0 = (rim1*rim1 + k2) / (rip1*rip1 + k2);
	}

	struct Event {
		vec3 reflectionDir;
		vec3 reflectance;
	};

	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;
		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;
		e.reflectance = r0 + (vec3(1, 1, 1) - r0) * pow(1 - cosa, 5);
		return e;
	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		//return normal;
		float w = position.x * period + 
			pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
		w -= int(w);
		return (vec3(1, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1 - w)) * normal.dot(viewDir);
	}

	vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		return powerDensity * getColor(position, hitNormal, viewDirection) 
			* (hitNormal.dot(lightDirection));

	}
};

class Glass : public Material
{
	float  refractiveIndex;
	float  r0;
public:
	Glass(float refractiveIndex) : refractiveIndex(refractiveIndex) {
		r0 = (refractiveIndex - 1)*(refractiveIndex - 1)
			/ (refractiveIndex + 1)*(refractiveIndex + 1);
	}
	struct Event {
		vec3 reflectionDir;
		vec3 refractionDir;
		float reflectance;
		float transmittance;
	};
	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;
		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;

		float ri = refractiveIndex;
		if (cosa < 0) { cosa = -cosa; normal = -normal; ri = 1 / ri; }
		float disc = 1 - (1 - cosa * cosa) / ri / ri;
		if (disc < 0)
			e.reflectance = 1;
		else {
			float cosb = sqrt(disc);
			e.refractionDir = parallel / ri - normal * cosb;
			e.reflectance = r0 + (1 - r0) * pow(1 - cosa, 5);
		}
		e.transmittance = 1 - e.reflectance;
		return e;
	}

	virtual vec3 shade(vec3 powerDensity, vec3 lightDirection,
		vec3 hitNormal, vec3 viewDirection, vec3 position) {
		vec3 h = (lightDirection + viewDirection).normalize();
		float temp = h.dot(hitNormal);
		return powerDensity * pow(temp, 21);
	}
};

class Metal :
	public Material
{
	vec3 r0;

public:
	Metal(vec3  refractiveIndex, vec3 extinctionCoefficient)
	{
		vec3 rim1 = refractiveIndex - vec3(1, 1, 1);
		vec3 rip1 = refractiveIndex + vec3(1, 1, 1);
		vec3 k2 = extinctionCoefficient * extinctionCoefficient;
		r0 = (rim1*rim1 + k2) / (rip1*rip1 + k2);
	}

	struct Event {
		vec3 reflectionDir;
		vec3 reflectance;
	};

	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;
		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;
		e.reflectance = r0 + (vec3(1, 1, 1) - r0) * pow(1 - cosa, 5);
		return e;
	}

	virtual vec3 shade(vec3 powerDensity, vec3 lightDirection, vec3 hitNormal, 
		vec3 viewDirection, vec3 position) {
		vec3 h  = (lightDirection + viewDirection).normalize();
		float temp = h.dot(hitNormal);
		return powerDensity * pow(temp, 21);
	}

};
// Camera class.
class Camera
{
	vec3 eye;		//< world space camera position
	vec3 lookAt;	//< center of window in world space
	vec3 right;		//< vector from window center to window right-mid (in world space)
	vec3 up;		//< vector from window center to window top-mid (in world space)

public:
	Camera()
	{
		eye = vec3(0, 0, 2);
		lookAt = vec3(0, 0, 1);
		right = vec3(1, 0, 0);
		up = vec3(0, 1, 0);
	}
	vec3 getEye()
	{
		return eye;
	}
	// compute ray through pixel at normalized device coordinates
	vec3 rayDirFromNdc(float x, float y) {
		return (lookAt - eye
			+ right * x
			+ up    * y
			).normalize();
	}
};

// Ray structure.
class Ray
{
public:
	vec3 origin;
	vec3 dir;
	Ray(vec3 o, vec3 d)
	{
		origin = o;
		dir = d;
	}
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
	Hit()
	{
		t = -1;
	}
	float t;				//< Ray paramter at intersection. Negative means no valid intersection.
	vec3 position;			//< Intersection coordinates.
	vec3 normal;			//< Surface normal at intersection.
	Material* material;		//< Material of intersected surface.
};

// Abstract base class.
class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material) :material(material) {}
	virtual Hit intersect(const Ray& ray) = 0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.
class QuadraticRoots
{
public:
	float t1;
	float t2;
	// Solves the quadratic a*t*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and sets members t1 and t2 to store the roots.
	QuadraticRoots(float a, float b, float c)
	{
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) // no roots
		{
			t1 = -1;
			t2 = -1;
			return;
		}
		float sqrt_discr = sqrt(discr);
		t1 = (-b + sqrt_discr) / 2.0 / a;
		t2 = (-b - sqrt_discr) / 2.0 / a;
	}
	// Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
	float getLesserPositive()
	{
		return (0 < t1 && t1 < t2) ? t1 : t2;
	}
};

// Object realization.
class Sphere : public Intersectable
{
	vec3 center;
	float radius;
public:
	Sphere(const vec3& center, float radius, Material* material) :
		Intersectable(material),
		center(center),
		radius(radius)
	{
	}
	QuadraticRoots solveQuadratic(const Ray& ray)
	{
		vec3 diff = ray.origin - center;
		float a = ray.dir.dot(ray.dir);
		float b = diff.dot(ray.dir) * 2.0;
		float c = diff.dot(diff) - radius * radius;
		return QuadraticRoots(a, b, c);

	}
	vec3 getNormalAt(vec3 r)
	{
		return (r - center).normalize();
	}
	Hit intersect(const Ray& ray)
	{
		// This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal
		float t = solveQuadratic(ray).getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
	}
};

// CLASS PLANE COULD COME HERE

class Plane : public Intersectable
{
	vec3 normal;
	vec3 point;
public:
	Plane(const vec3& normal, const vec3& point, Material* material) :
		Intersectable(material),
		normal(normal),
		point(point)
	{ }

	Hit intersect(const Ray& ray)
	{
		float t = (point - ray.origin).dot(normal) / (ray.dir.dot(normal));

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = normal;

		return hit;
	}
};

// CLASS QUADRIC COULD COME HERE

class Quadric :
	public Intersectable
{
	mat4x4 coeffs;

public:
	Quadric(Material* material) :
		Intersectable(material)
	{
		coeffs = mat4x4();
		coeffs._00 = 1;
		coeffs._11 = 1;
		coeffs._22 = 1;
		coeffs._33 = -1;

	}

	Quadric* cylinder() {
		coeffs._11 = 0;
		return this;
	}

	Quadric* cone() {
		coeffs._11 = -1;
		coeffs._33 = 0;
		//coeffs._12 = 1;
		return this;
	}

	Quadric* paraboloid() {
		coeffs._11 = 0;
		coeffs._33 = 0;
		coeffs._13 = -1;
		return this;
	}

	Quadric* hyperboloid() {
		coeffs._11 = -1;
		return this;
	}

	Quadric* parallelPlanes() {
		coeffs._00 = 0;
		coeffs._11 = 1;
		coeffs._22 = 0;
		coeffs._33 = -1;
		return this;

	}

	Quadric* transform(mat4x4 t) {
		mat4x4 ti = t.invert();
		coeffs = ti * (coeffs * ti.transpose());
		return this;
	}

	QuadraticRoots solveQuadratic(const Ray& ray)
	{
		vec4 e = vec4(ray.origin);
		vec4 d = vec4(ray.dir);
		d.w = 0;
		float a = d.dot(coeffs * d);
		float b = e.dot(coeffs * d) + d.dot(coeffs * e);
		float c = e.dot(coeffs * e);
		return QuadraticRoots(a, b, c);

	}

	vec3 getNormalAt(vec3 r)
	{
		vec4 r4 = vec4(r);
		vec4 prenormal = (coeffs * r) + operator*(r, coeffs);
		return vec3(prenormal.x, prenormal.y, prenormal.z).normalize();
	}
	Hit intersect(const Ray& ray)
	{
		float t = solveQuadratic(ray).getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
	}

	bool contains(vec3 r) {
		vec4 rhomo(r);
		float hitTry = rhomo.dot(coeffs * rhomo);
		if (hitTry > 0) {
			return false;
		}
		else {
			return true;
		}

	}

};
// CLASS CLIPPEDQUADRIC COULD COME HERE

class ClippedQuadric : public Intersectable {

	Quadric* shape;
	Quadric* clipper;

public:
	ClippedQuadric(Quadric* shape, Quadric* clipper, Material* material) :
		shape(shape),
		clipper(clipper),
		Intersectable(material)
	{
	}

	Hit intersect(const Ray& ray)
	{
		QuadraticRoots roots = shape->solveQuadratic(ray);

		Hit hit1;
		hit1.t = roots.t1;
		hit1.material = material;
		hit1.position = ray.origin + ray.dir * roots.t1;
		hit1.normal = shape->getNormalAt(hit1.position);

		Hit hit2;
		hit2.t = roots.t2;
		hit2.material = material;
		hit2.position = ray.origin + ray.dir * roots.t2;
		hit2.normal = shape->getNormalAt(hit2.position);

		if (!clipper->contains(hit1.position)) {
			hit1.t = -1;
		}
		if (!clipper->contains(hit2.position)) {
			hit2.t = -1;
		}

		if (hit1.t < 0) {
			return hit2;
		}
		if (hit2.t < 0) {
			return hit1;
		}
		if (hit1.t < hit2.t)
			return hit1;
		else
			return hit2;

	}

};

class LightSource
{

public:

	virtual vec3 getPowerDensityAt(vec3 x) = 0;
	virtual vec3 getLightDirAt(vec3 x) = 0;
	virtual float  getDistanceFrom(vec3 x) = 0;
};

class DirectionalLight : public LightSource
{
	vec3 direction;
	vec3 powerDensity;

public:
	DirectionalLight(vec3 powerDensitya, vec3 directiona) :
		LightSource(),
		direction(direction)
	{
		direction = directiona;
		powerDensity = powerDensitya;
	}

	vec3 getPowerDensityAt(vec3 x) {
		return powerDensity;
	} 

	vec3 getLightDirAt(vec3 x) {
		return direction;
	}

	float  getDistanceFrom(vec3 x) {
		return 1000000;
	}
};

class PointLight : public LightSource
{
	vec3 position;
	vec3 powerDensity;

public:
	PointLight(vec3 powerDensitya, vec3 positiona) :
		LightSource(),
		position(position)
	{
		powerDensity = powerDensitya;
		position = positiona; 
	}

	vec3 getPowerDensityAt(vec3 x) {
		return powerDensity * (1.0 / (float)(4.0 * 3.14159 * (position - x).  norm2()));
	}

	vec3 getLightDirAt(vec3 x) {
		return (position - x).normalize();
	}

	float getDistanceFrom(vec3 x) {
		return (position - x).norm();
	}

};

class Scene
{
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
	std::vector<LightSource*> lights;
	int maxDepth = 2;

public:
	Scene()
	{
		materials.push_back(new ReflectiveWood(vec3(.21, .485, 1.29), vec3(0, 0, 0)));
		materials.push_back(new DiffuseMaterial(vec3(0,1,0)));
		materials.push_back(new DiffuseMaterial(vec3(1, 1, 1)));
		materials.push_back(new DiffuseMaterial(vec3(1, .25, 0)));
		materials.push_back(new DiffuseMaterial(vec3(.2, .2, .2)));
		materials.push_back(new Glass(.87));
		materials.push_back(new Metal(vec3(.21, .485, 1.29), vec3(0, 0, 1)));
		materials.push_back(new Wood());
		materials.push_back(new EmissiveTwo(vec3(1, .25, 0)));
		materials.push_back(new Metal(vec3(.21, .485, 1.29), vec3(1, 0, 0)));
		materials.push_back(new Metal(vec3(.21, .485, 1.29), vec3(0, 1, 0)));
		materials.push_back(new Emissive(vec3(1, 1, 1)));
		materials.push_back(new Metal(vec3(.21, .485, 1.29), vec3(1, 1, 0)));

		lights.push_back(new DirectionalLight(vec3(.15, .15, .15), vec3(0, 2, 2)));
		lights.push_back(new PointLight(vec3(30, 30, 30), vec3(0, -.2, -.3)));

		objects.push_back(new Plane(vec3(0, 1, 0), vec3(0, -2, 0), materials[0]));
		
		Quadric* tree1 = new Quadric(materials[1]);
		tree1->cone();
		tree1->transform(mat4x4::translation(vec3(1.5, 0, -1)));

		Quadric* clip1 = new Quadric(materials[1]);
		clip1->parallelPlanes();
		clip1->transform(mat4x4::scaling(vec3(1, .2, 1)));
		clip1->transform(mat4x4::translation(vec3(1.5, -.2, -1)));

		Quadric* tree2 = new Quadric(materials[1]);
		tree2->cone();
		tree2->transform(mat4x4::translation(vec3(1.5, -.2, -1)));

		Quadric* clip2 = new Quadric(materials[1]);
		clip2->parallelPlanes();
		clip2->transform(mat4x4::scaling(vec3(1, .3, 1)));
		clip2->transform(mat4x4::translation(vec3(1.5, -.5, -1)));

		Quadric* tree3 = new Quadric(materials[1]);
		tree3->cone();
		tree3->transform(mat4x4::translation(vec3(1.5, -.5, -1)));

		Quadric* clip3 = new Quadric(materials[1]);
		clip3->parallelPlanes();
		clip3->transform(mat4x4::scaling(vec3(1, .3, 1)));
		clip3->transform(mat4x4::translation(vec3(1.5, -.9, -1)));

		Quadric* tree4 = new Quadric(materials[1]);
		tree4->cylinder();
		tree4->transform(mat4x4::scaling(vec3(.2, .2, .2)));
		tree4->transform(mat4x4::translation(vec3(1.5, -.2, -1)));

		Quadric* clip4 = new Quadric(materials[1]);
		clip4->parallelPlanes();
		clip4->transform(mat4x4::scaling(vec3(1, 1, 1)));
		clip4->transform(mat4x4::translation(vec3(1.5, -2, -1)));

		Quadric* snow1 = new Quadric(materials[2]);
		snow1->transform(mat4x4::translation(vec3(-2, -7, -1)));
		snow1->transform(mat4x4::scaling(vec3(2, .3, 2)));

		Quadric* snow2 = new Quadric(materials[2]);
		snow2->transform(mat4x4::translation(vec3(-1.3, -6.8, -.5)));
		snow2->transform(mat4x4::scaling(vec3(2, .3, 2)));

		Quadric* candle = new Quadric(materials[1]);
		candle->cylinder();
		candle->transform(mat4x4::scaling(vec3(.1, .07, .1)));
		candle->transform(mat4x4::translation(vec3(0, -.03, -.6)));

		Quadric* candleclip = new Quadric(materials[1]);
		candleclip->parallelPlanes();
		candleclip->transform(mat4x4::scaling(vec3(1, .8, 1)));
		candleclip->transform(mat4x4::translation(vec3(1, -2, -1)));

		Quadric* flame = new Quadric(materials[8]);
		flame->transform(mat4x4::scaling(vec3(.05, .2, .1)));
		flame->transform(mat4x4::translation(vec3(0, -1.3, -.6)));

		Quadric* nose = new Quadric(materials[3]);
		nose->cone();
		nose->transform(mat4x4::rotation(vec3(1, 0, 1), 90));
		nose->transform(mat4x4::rotation(vec3(0, 1, 0), 140));
		nose->transform(mat4x4::scaling(vec3(.3, .3, .3)));
		nose->transform(mat4x4::translation(vec3(-1.3, -.92, -.38)));

		Quadric* noseclip = new Quadric(materials[1]);
		noseclip->parallelPlanes();
		noseclip->transform(mat4x4::scaling(vec3(.05, .05, .05)));
		noseclip->transform(mat4x4::rotation(vec3(1, 0, 1), 90));
		noseclip->transform(mat4x4::rotation(vec3(0, 1, 0), 140));
		noseclip->transform(mat4x4::translation(vec3(-1, -.85, -.85)));

		objects.push_back(new ClippedQuadric(tree1, clip1, materials[1]));
		objects.push_back(new ClippedQuadric(tree2, clip2, materials[1]));
		objects.push_back(new ClippedQuadric(tree3, clip3, materials[1]));
		objects.push_back(new ClippedQuadric(tree4, clip4, materials[7]));

		objects.push_back(snow1);
		objects.push_back(snow2);
		objects.push_back(new Sphere(vec3(-1.5, -1.6, -.5), .3, materials[2]));
		objects.push_back(new Sphere(vec3(-1.5, -1.2, -.5), .25, materials[2]));
		objects.push_back(new Sphere(vec3(-1.5, -.85, -.5), .2, materials[2]));

		objects.push_back(new ClippedQuadric(candle, candleclip, materials[2]));
		objects.push_back(flame);

		objects.push_back(new ClippedQuadric(nose, noseclip, materials[3]));
		objects.push_back(new Sphere(vec3(-1.3, -1.05, -.42), .03, materials[4]));
		objects.push_back(new Sphere(vec3(-1.28, -1.2, -.4), .03, materials[4]));

		objects.push_back(new Sphere(vec3(-1.43, -.8, -.33), .03, materials[4]));
		objects.push_back(new Sphere(vec3(-1.3, -.8, -.5), .03, materials[4]));

		Quadric* ice1 = new Quadric(materials[2]);
		ice1->cone();
		ice1->transform(mat4x4::scaling(vec3(.05, .3, .05)));
		ice1->transform(mat4x4::translation(vec3(-3.5, 1, -3)));

		Quadric* icecli1 = new Quadric(materials[1]); 
		icecli1->parallelPlanes();
		icecli1->transform(mat4x4::scaling(vec3(1, 2, 1)));
		icecli1->transform(mat4x4::translation(vec3(0, 3, 0)));

		Quadric* ice2 = new Quadric(materials[2]);
		ice2->cone();
		ice2->transform(mat4x4::scaling(vec3(.05, .3, .05)));
		ice2->transform(mat4x4::translation(vec3(-4.5, .5, -3)));

		Quadric* icecli2 = new Quadric(materials[1]);
		icecli2->parallelPlanes();
		icecli2->transform(mat4x4::scaling(vec3(1, 2, 1)));
		icecli2->transform(mat4x4::translation(vec3(0, 2.5, 0)));

		Quadric* ice3 = new Quadric(materials[2]);
		ice3->cone();
		ice3->transform(mat4x4::scaling(vec3(.05, .3, .05)));
		ice3->transform(mat4x4::translation(vec3(2.5, 1, -3)));

		Quadric* icecli3 = new Quadric(materials[1]);
		icecli3->parallelPlanes();
		icecli3->transform(mat4x4::scaling(vec3(1, 2, 1)));
		icecli3->transform(mat4x4::translation(vec3(0, 3, 0)));

		objects.push_back(new ClippedQuadric(ice1, icecli1, materials[5]));
		objects.push_back(new ClippedQuadric(ice2, icecli2, materials[5]));
		objects.push_back(new ClippedQuadric(ice3, icecli3, materials[5]));

		objects.push_back(new Sphere(vec3(1, -1, -.8), .05, materials[6]));
		objects.push_back(new Sphere(vec3(1.25, -1.05, -.5), .05, materials[9]));
		objects.push_back(new Sphere(vec3(1.6, -1, -.5), .05, materials[10]));
		objects.push_back(new Sphere(vec3(1.15, -.6, -.75), .05, materials[9]));
		objects.push_back(new Sphere(vec3(1.45, -.6, -.6), .05, materials[6]));
		objects.push_back(new Sphere(vec3(1.33, -.25, -.75), .05, materials[10]));

		Quadric* bell1 = new Quadric(materials[2]);
		bell1->paraboloid();
		bell1->transform(mat4x4::rotation(vec3(1, 0, 0), 179));
		bell1->transform(mat4x4::scaling(vec3(.25, .25, .25)));
		bell1->transform(mat4x4::translation(vec3(0, 1.2, -2.1)));

		Quadric* bell2 = new Quadric(materials[2]);
		bell2->parallelPlanes();
		bell2->transform(mat4x4::scaling(vec3(.5, .5, .5)));
		bell2->transform(mat4x4::translation(vec3(0, 1, 0)));

		Quadric* bell3 = new Quadric(materials[3]);
		bell3->hyperboloid();
		bell3->transform(mat4x4::scaling(vec3(.3, .5, .3)));
		bell3->transform(mat4x4::translation(vec3(0, 1, -2)));

		Quadric* bell4 = new Quadric(materials[2]);
		bell4->parallelPlanes();
		bell4->transform(mat4x4::scaling(vec3(.5, .3, .5)));
		bell4->transform(mat4x4::translation(vec3(0, .5, 0)));

		objects.push_back(new ClippedQuadric(bell1, bell2, materials[12]));
		objects.push_back(new ClippedQuadric(bell3, bell4, materials[12]));

		double j = .8;
		double a;
		double b;
		double c;

		for (double i = -3.5; i < 3.5; i = i + .4) {
			a = (rand() % 100 + 70) / 100;
			b = (rand() % 100 + 70) / 100;
			c = (rand() % 100 + 70) / 100;
			objects.push_back(new Sphere(vec3(i, j, -2.1), .05, 
				new Emissive(vec3(a, b, c))));
			lights.push_back(new PointLight(vec3(10, 10, 10), vec3(i, j, -2)));
			j = j + .05;
		}


		//objects.push_back(new Sphere(vec3(-2, 1, -.1), .5, materials[6]));

	}
	~Scene()
	{
		// UNCOMMENT THESE WHEN APPROPRIATE
		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
			delete *iMaterial;
		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
			delete *iObject;		
	} 

public:
	Camera& getCamera()
	{
		return camera;
	}

	Hit firstIntersect(const Ray& ray) {

		Hit first;
		Hit temp;
		float t = FLT_MAX;

		for (unsigned int i = 0; i < objects.size(); i++) {
			temp = objects[i]->intersect(ray);
			if (temp.t >= 0.0 && temp.t < t) {
				first = temp;
				t = temp.t;
			}
		}

		return first;
	}

	float scale = 10;
	float turbulence = 100;
	float period = 1;
	float sharpness = 6;

	float snoiseA(vec3 r) {
		unsigned int x = 0x0625DF73;
		unsigned int y = 0xD1B84B45;
		unsigned int z = 0x152AD8D0;
		float f = 0;
		for (int i = 0; i<32; i++) {
			vec3 s(x / (float)0xffffffff,
				y / (float)0xffffffff,
				z / (float)0xffffffff);
			f += sin(s.dot(r));
			x = x << 1 | x >> 31;
			y = y << 1 | y >> 31;
			z = z << 1 | z >> 31;
		}
		return f / 64.0 + 0.5;
	}

	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		//return normal;
		float w = position.x * period + pow(snoiseA(position * scale), sharpness)
			*turbulence + 10000.0;
		w -= int(w);
		return (vec3(.3, .5, 1) * w + vec3(.5, 1, .5) * (1 - w));
	}

	vec3 trace(const Ray& ray, int depth)
	{

		vec3 color = vec3(0.f, 0.f, 0.f);
		Hit hit = firstIntersect(ray);
		float offset = 0.0001;

		if (hit.t < 0)
		{
			return getColor(ray.dir, vec3(0,0,0), vec3(0, 0, 0));
		}

		Emissive* emissive = dynamic_cast<Emissive*>(hit.material);

		if (emissive != NULL) {
			return hit.material->shade(
				vec3(1, 1, 1),
				vec3(0, -1, 1),
				hit.normal,
				-ray.dir,
				hit.position
			);
		}

		for (std::size_t i = 0; i < lights.size(); i++) {
			Ray shadowRay(hit.position + hit.normal * offset,
				lights[i]->getLightDirAt(hit.position));
			Hit shadowHit = firstIntersect(shadowRay);

			Emissive* shadowemissive = dynamic_cast<Emissive*>(shadowHit.material);

			if (shadowHit.t < 0 || shadowemissive != NULL) {
				color += hit.material->shade(
					lights[i]->getPowerDensityAt(hit.position),
					lights[i]->getLightDirAt(hit.position),
					hit.normal,
					-ray.dir,
					hit.position
				);
			}
		}

		if (depth > maxDepth)
			return vec3(0, 0, 0);

		Glass* glass = dynamic_cast<Glass*>(hit.material);
		if (glass != NULL) {
			Glass::Event e = glass->evaluateEvent(ray.dir, hit.normal);
			color += trace(Ray(hit.position + hit.normal, e.reflectionDir), depth++) * e.reflectance;
			if (e.transmittance > 0)
				color += trace(Ray(hit.position - hit.position, e.refractionDir), depth++) * e.transmittance;
		}

		Metal* metal = dynamic_cast<Metal*>(hit.material);
		if (metal != NULL) {
			Metal::Event e = metal->evaluateEvent(ray.dir, hit.normal);
			color += trace(Ray(hit.position + hit.normal, e.reflectionDir), depth++) * e.reflectance;
		}

		ReflectiveWood* wood = dynamic_cast<ReflectiveWood*>(hit.material);
		if (wood != NULL) {
			ReflectiveWood::Event e = wood->evaluateEvent(ray.dir, hit.normal);
			color += trace(Ray(hit.position + hit.normal, e.reflectionDir), depth++) * e.reflectance;
		}


		return color;
	}

};


Scene scene;

bool computeImage()
{
	static unsigned int iPart = 0;

	if (iPart >= 64)
		return false;
	for (int j = iPart; j < windowHeight; j += 64)
	{
		for (int i = 0; i < windowWidth; i++)
		{
			float ndcX = (2.0 * i - windowWidth) / windowWidth;
			float ndcY = (2.0 * j - windowHeight) / windowHeight;
			Camera& camera = scene.getCamera();
			Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcX, ndcY));

			image[j*windowWidth + i] = scene.trace(ray, 0);
		}
	}
	iPart++;
	return true;
}

void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (computeImage())
		glutPostRedisplay();
	glDrawPixels(windowWidth, windowHeight, GL_RGB, GL_FLOAT, image);

	glutSwapBuffers();
}



int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(0, 0);
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Ray Casting");

#if !defined(__APPLE__)
	glewExperimental = true;
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	glViewport(0, 0, windowWidth, windowHeight);

	glutDisplayFunc(onDisplay);

	glutMainLoop();

	return 1;
}
