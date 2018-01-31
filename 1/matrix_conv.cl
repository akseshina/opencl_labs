inline size_t to_1D(size_t y, size_t x, size_t n) {
    return y * n + x;
}


bool valid(int i, int n) {
    return 0 <= i && i < n;
}


__kernel void convolution(__global float * A,
                          __global float * B,
                          __global float * C,
                          int n, int m)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float res = 0;
    int hm = (m - 1) / 2;

    for (int k = -hm; k <= hm; k++)
        for (int p = -hm; p <= hm; p++)
            if (valid(row + k, n) && valid(col + p, n)) {
                int i_a = to_1D(row + k, col + p, n);
                int i_b = to_1D(k + hm, p + hm, m);
                res += A[i_a] * B[i_b];
            }

    C[to_1D(row, col, n)] = res;
}