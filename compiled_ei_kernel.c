#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Python min/max retain their first argument for ties and NaNs. */
static double mn(double a, double b) { return b < a ? b : a; }
static double mx(double a, double b) { return b > a ? b : a; }
static double (*volatile faithful_pow)(double, double) = pow;

static double checked_exp(double x) {
    double value = exp(x);
    if (isfinite(x) && isinf(value))
        PyErr_SetString(PyExc_OverflowError, "math range error");
    return value;
}
static double checked_log(double x) {
    if (x <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "math domain error");
        return NAN;
    }
    return log(x);
}
static double logdiff(double x, double y) {
    double delta = mn(y - x, nextafter(0.0, -INFINITY));
    /* Python's final x + value preserves x's NaN sign/payload. GCC may
     * otherwise choose the other NaN operand even without fast-math. */
    if (isnan(x)) return x;
    if (delta < log(0.5)) return x + log1p(-exp(delta));
    return x + log(-expm1(delta));
}
static double positive(double lower, double upper, double log_mass) {
    double width = upper - lower;
    double scaled_span = lower * width + 0.5 * width * width;
    double aw, width2, mass_sum, moment_sum, previous, term, mass_add, moment_add, following;
    double inverse, inverse2, series, density_lower, density_upper;
    int order;
    if (scaled_span <= 4.0) {
        aw = lower * width; width2 = width * width;
        mass_sum = 1.0; moment_sum = 0.5; previous = 1.0; term = -aw;
        for (order = 1; order < 100; order++) {
            mass_add = term / (order + 1); moment_add = term / (order + 2);
            mass_sum += mass_add; moment_sum += moment_add;
            if (order >= 4 && fabs(term) <= 2e-16 * mx(fabs(mass_sum), 1.0)
                && fabs(previous) <= 2e-16 * mx(fabs(moment_sum), 1.0)) break;
            following = -(aw * term + width2 * previous) / (order + 1);
            previous = term; term = following;
        }
        if (mass_sum == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
            return NAN;
        }
        return mn(width, mx(0.0, width * moment_sum / mass_sum));
    }
    if (lower >= 20.0 && scaled_span >= 36.0) {
        inverse = 1.0 / lower; inverse2 = inverse * inverse;
        series = 1.0 - 2.0 * inverse2 + 10.0 * faithful_pow(inverse2, 2.0)
            - 74.0 * faithful_pow(inverse2, 3.0) + 706.0 * faithful_pow(inverse2, 4.0)
            - 8162.0 * faithful_pow(inverse2, 5.0) + 110410.0 * faithful_pow(inverse2, 6.0);
        return mn(width, mx(0.0, inverse * series));
    }
    density_lower = checked_exp(-0.5 * lower * lower - 0.5 * log(2.0 * 3.141592653589793) - log_mass);
    if (PyErr_Occurred()) return NAN;
    density_upper = checked_exp(-0.5 * upper * upper - 0.5 * log(2.0 * 3.141592653589793) - log_mass);
    return mn(width, mx(0.0, density_lower - density_upper - lower));
}
static double excess(double lower, double upper, double log_mass) {
    double reflected, density_lower, density_upper, mean;
    if (lower >= 0.0) return positive(lower, upper, log_mass);
    if (upper <= 0.0) {
        reflected = positive(-upper, -lower, log_mass);
        return mn(upper - lower, mx(0.0, upper - lower - reflected));
    }
    density_lower = checked_exp(-0.5 * lower * lower - 0.5 * log(2.0 * 3.141592653589793) - log_mass);
    if (PyErr_Occurred()) return NAN;
    density_upper = checked_exp(-0.5 * upper * upper - 0.5 * log(2.0 * 3.141592653589793) - log_mass);
    mean = density_lower - density_upper;
    return mn(upper - lower, mx(0.0, mean - lower));
}
/* Private API: equally sized, aligned, native-double 1D buffers, read-only OK.
 * The GIL and borrowed views remain held; inputs are never written. */
static int buffers(PyObject *args, Py_buffer *views, int n) {
    int i;
    if (PyTuple_Size(args) != n) { PyErr_SetString(PyExc_TypeError, "buffer count"); return 0; }
    for (i = 0; i < n; i++) {
        if (PyObject_GetBuffer(PyTuple_GetItem(args, i), &views[i], PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) break;
        if (views[i].itemsize != sizeof(double) || views[i].ndim != 1
            || !views[i].format || (strcmp(views[i].format, "d") && strcmp(views[i].format, "@d") && strcmp(views[i].format, "=d"))
            || views[i].len < 0 || views[i].len % sizeof(double)
            || !views[i].shape || views[i].shape[0] != views[i].len / (Py_ssize_t)sizeof(double)
            || !PyBuffer_IsContiguous(&views[i], 'C')
            || (views[i].len && (uintptr_t)views[i].buf % _Alignof(double))
            || (i && views[i].len != views[0].len)) {
            PyBuffer_Release(&views[i]);
            PyErr_SetString(PyExc_ValueError, "aligned native float64 contiguous 1D buffers of equal length required");
            break;
        }
    }
    if (i == n) return 1;
    while (i--) PyBuffer_Release(&views[i]);
    return 0;
}
static PyObject *mass(PyObject *self, PyObject *args) {
    Py_buffer v[3]; Py_ssize_t i, n; const double *a, *b, *kind; PyObject *out; double value;
    if (!buffers(args, v, 3)) return NULL;
    n = v[0].len / sizeof(double); a = v[0].buf; b = v[1].buf; kind = v[2].buf;
    out = PyBytes_FromStringAndSize(NULL, v[0].len);
    if (out) {
        for (i = 0; i < n; i++) {
            value = kind[i] ? logdiff(a[i], b[i]) : checked_log(0.5 * (a[i] + b[i]));
            if (PyErr_Occurred()) { Py_CLEAR(out); break; }
            memcpy(PyBytes_AS_STRING(out) + i * sizeof(double), &value, sizeof(double));
        }
    }
    for (i = 0; i < 3; i++) PyBuffer_Release(&v[i]);
    return out;
}
static PyObject *inner(PyObject *self, PyObject *args) {
    Py_buffer v[8]; Py_ssize_t i, n; const double *a[8]; double ratio, tail_excess, value; PyObject *out;
    if (!buffers(args, v, 8)) return NULL;
    n = v[0].len / sizeof(double);
    for (i = 0; i < 8; i++) a[i] = v[i].buf;
    out = PyBytes_FromStringAndSize(NULL, v[0].len);
    if (out) {
        for (i = 0; i < n; i++) {
            ratio = exp(mn(0.0, a[1][i] - a[0][i]));
            tail_excess = a[2][i] * excess(a[3][i], a[4][i], a[1][i]);
            if (PyErr_Occurred()) { Py_CLEAR(out); break; }
            value = mx(a[6][i] - a[5][i], 0.0) + ratio * tail_excess;
            value = mn(mx(a[7][i] - a[5][i], 0.0), mx(value, 0.0));
            memcpy(PyBytes_AS_STRING(out) + i * sizeof(double), &value, sizeof(double));
        }
    }
    for (i = 0; i < 8; i++) PyBuffer_Release(&v[i]);
    return out;
}
static PyMethodDef methods[] = {{"mass", mass, METH_VARARGS, "Private scalar log mass loop."}, {"inner", inner, METH_VARARGS, "Private excess loop; caller supplies matching mass."}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_compiled_ei_kernel", NULL, -1, methods};
PyMODINIT_FUNC PyInit__compiled_ei_kernel(void) { return PyModule_Create(&module); }
