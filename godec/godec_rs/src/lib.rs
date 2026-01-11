use fwht::fwht_mut;
use nalgebra::DMatrix;
use numpy::{ndarray::Array2, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::index;
use rand::Rng;
use rand_distr::StandardNormal;
use rustfft::{num_complex::Complex, FftPlanner};

#[cfg(feature = "blas")]
use cblas::{dgemm, Layout, Transpose as BlasTranspose};
#[cfg(feature = "blas")]
use openblas_src as _;

#[cfg(not(feature = "blas"))]
#[derive(Clone, Copy)]
enum BlasTranspose {
    None,
    Ordinary,
}

#[derive(Clone, Copy)]
enum ProjectionKind {
    Gaussian,
    Rademacher,
    Achlioptas,
    SparseJl,
    Srht,
    Srft,
    BlockSparse,
}

impl ProjectionKind {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gaussian" => Some(Self::Gaussian),
            "rademacher" => Some(Self::Rademacher),
            "achlioptas" => Some(Self::Achlioptas),
            "sparse_jl" | "sparsejl" | "sjlt" => Some(Self::SparseJl),
            "srht" | "fjlt" => Some(Self::Srht),
            "srft" => Some(Self::Srft),
            "block_sparse" | "blocksparse" => Some(Self::BlockSparse),
            _ => None,
        }
    }
}

fn next_pow2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}

fn hadamard_inplace(data: &mut [f64]) {
    fwht_mut(data).expect("hadamard input length must be power of two");
}

fn generate_projection(
    n: usize,
    rank: usize,
    kind: ProjectionKind,
    rng: &mut impl Rng,
) -> DMatrix<f64> {
    match kind {
        ProjectionKind::Gaussian => DMatrix::from_fn(n, rank, |_, _| rng.sample(StandardNormal)),
        ProjectionKind::Rademacher => DMatrix::from_fn(n, rank, |_, _| {
            if rng.gen::<bool>() { 1.0 } else { -1.0 }
        }),
        ProjectionKind::Achlioptas => DMatrix::from_fn(n, rank, |_, _| {
            let r: f64 = rng.gen();
            if r < 1.0 / 6.0 {
                (3.0f64).sqrt()
            } else if r < 5.0 / 6.0 {
                0.0
            } else {
                -(3.0f64).sqrt()
            }
        }),
        ProjectionKind::SparseJl => {
            let mut data = vec![0.0f64; n * rank];
            let s = (n / 50).max(1);
            let scale = 1.0 / (s as f64).sqrt();
            for col in 0..rank {
                let sample = index::sample(rng, n, s.min(n));
                for idx in sample.iter() {
                    let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                    data[idx * rank + col] = sign * scale;
                }
            }
            DMatrix::from_row_slice(n, rank, &data)
        }
        ProjectionKind::Srht => {
            let size = next_pow2(n);
            let scale = 1.0 / (size as f64).sqrt();
            let mut data = vec![0.0f64; n * rank];
            for col in 0..rank {
                let mut v = vec![0.0f64; size];
                for i in 0..n {
                    v[i] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                }
                hadamard_inplace(&mut v);
                for i in 0..n {
                    data[i * rank + col] = v[i] * scale;
                }
            }
            DMatrix::from_row_slice(n, rank, &data)
        }
        ProjectionKind::Srft => {
            let size = next_pow2(n);
            let scale = 1.0 / (size as f64).sqrt();
            let mut planner = FftPlanner::<f64>::new();
            let fft = planner.plan_fft_forward(size);
            let mut data = vec![0.0f64; n * rank];
            for col in 0..rank {
                let mut v: Vec<Complex<f64>> = (0..size)
                    .map(|i| {
                        let sign = if i < n && rng.gen::<bool>() { 1.0 } else if i < n { -1.0 } else { 0.0 };
                        Complex::new(sign, 0.0)
                    })
                    .collect();
                fft.process(&mut v);
                for i in 0..n {
                    data[i * rank + col] = v[i].re * scale;
                }
            }
            DMatrix::from_row_slice(n, rank, &data)
        }
        ProjectionKind::BlockSparse => {
            let block = 256usize.min(n.max(1));
            let scale = 1.0 / (block as f64).sqrt();
            let mut data = vec![0.0f64; n * rank];
            for col in 0..rank {
                let start = if n > block { rng.gen_range(0..=(n - block)) } else { 0 };
                for i in 0..block {
                    let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                    data[(start + i) * rank + col] = sign * scale;
                }
            }
            DMatrix::from_row_slice(n, rank, &data)
        }
    }
}

fn sketch_sparse_jl(l: &DMatrix<f64>, rank: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let (m, n) = l.shape();
    let s = (n / 50).max(1);
    let scale = 1.0 / (s as f64).sqrt();
    let mut y1 = DMatrix::<f64>::zeros(m, rank);
    for col in 0..rank {
        let sample = index::sample(rng, n, s.min(n));
        for idx in sample.iter() {
            let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            for i in 0..m {
                y1[(i, col)] += sign * scale * l[(i, idx)];
            }
        }
    }
    y1
}

fn sketch_block_sparse(l: &DMatrix<f64>, rank: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let (m, n) = l.shape();
    let block = 256usize.min(n.max(1));
    let scale = 1.0 / (block as f64).sqrt();
    let mut y1 = DMatrix::<f64>::zeros(m, rank);
    for col in 0..rank {
        let start = if n > block { rng.gen_range(0..=(n - block)) } else { 0 };
        for j in 0..block {
            let idx = start + j;
            let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            for i in 0..m {
                y1[(i, col)] += sign * scale * l[(i, idx)];
            }
        }
    }
    y1
}

fn sketch_srht(l: &DMatrix<f64>, rank: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let (m, n) = l.shape();
    let size = next_pow2(n);
    let sample = index::sample(rng, size, rank.min(size));
    let scale = 1.0 / (rank as f64).sqrt();
    let mut y1 = DMatrix::<f64>::zeros(m, rank);
    let mut signed = vec![0.0f64; size];

    let mut signs = vec![1.0f64; size];
    for s in signs.iter_mut().take(n) {
        *s = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    }

    for row in 0..m {
        signed.fill(0.0);
        for col in 0..n {
            signed[col] = l[(row, col)] * signs[col];
        }
        hadamard_inplace(&mut signed);
        for (k, idx) in sample.iter().enumerate() {
            y1[(row, k)] = signed[idx] * scale;
        }
    }
    y1
}

fn sketch_srft(l: &DMatrix<f64>, rank: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let (m, n) = l.shape();
    let size = next_pow2(n);
    let sample = index::sample(rng, size, rank.min(size));
    let scale = 1.0 / (rank as f64).sqrt();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(size);
    let mut y1 = DMatrix::<f64>::zeros(m, rank);

    let mut signs = vec![1.0f64; size];
    for s in signs.iter_mut().take(n) {
        *s = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    }

    let mut buf = vec![Complex::new(0.0f64, 0.0f64); size];
    for row in 0..m {
        for i in 0..size {
            buf[i] = Complex::new(0.0, 0.0);
        }
        for col in 0..n {
            buf[col] = Complex::new(l[(row, col)] * signs[col], 0.0);
        }
        fft.process(&mut buf);
        for (k, idx) in sample.iter().enumerate() {
            y1[(row, k)] = buf[idx].re * scale;
        }
    }
    y1
}

fn gemm(a: &DMatrix<f64>, ta: BlasTranspose, b: &DMatrix<f64>, tb: BlasTranspose) -> DMatrix<f64> {
    let (ar, ac) = a.shape();
    let (br, bc) = b.shape();
    let (m, k_a) = match ta {
        BlasTranspose::None => (ar, ac),
        _ => (ac, ar),
    };
    let (k_b, n) = match tb {
        BlasTranspose::None => (br, bc),
        _ => (bc, br),
    };
    assert_eq!(k_a, k_b);
    let mut c = DMatrix::<f64>::zeros(m, n);

    let lda = ar as i32;
    let ldb = br as i32;
    let ldc = m as i32;
    #[cfg(feature = "blas")]
    unsafe {
        dgemm(
            Layout::ColumnMajor,
            ta,
            tb,
            m as i32,
            n as i32,
            k_a as i32,
            1.0,
            a.as_slice(),
            lda,
            b.as_slice(),
            ldb,
            0.0,
            c.as_mut_slice(),
            ldc,
        );
        c
    }

    #[cfg(not(feature = "blas"))]
    {
        let a_mat = match ta {
            BlasTranspose::None => a.clone(),
            BlasTranspose::Ordinary => a.transpose(),
        };
        let b_mat = match tb {
            BlasTranspose::None => b.clone(),
            BlasTranspose::Ordinary => b.transpose(),
        };
        a_mat * b_mat
    }
}

fn dmatrix_from_pyarray(x: PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let array = x.as_array();
    let owned = array.to_owned();
    let owned = owned.as_standard_layout().to_owned();
    let slice = owned.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array must be contiguous")
    })?;
    let (rows, cols) = owned.dim();
    Ok(DMatrix::from_row_slice(rows, cols, slice))
}

fn dmatrix_to_pyarray(py: Python<'_>, m: &DMatrix<f64>) -> Py<PyArray2<f64>> {
    let (rows, cols) = m.shape();
    let mut data = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            data[i * cols + j] = m[(i, j)];
        }
    }
    let array = Array2::from_shape_vec((rows, cols), data).expect("shape matches data length");
    PyArray2::from_owned_array(py, array).to_owned()
}

fn rmse(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    let mut sum = 0.0f64;
    let len = a.len() as f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    (sum / len).sqrt()
}

fn godec_core(
    mut x: DMatrix<f64>,
    rank: usize,
    card: usize,
    iterated_power: usize,
    max_iter: usize,
    tol: f64,
    projection: ProjectionKind,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    if x.nrows() < x.ncols() {
        x = x.transpose();
    }

    let (m, n) = x.shape();
    let mut l = x.clone();
    let mut s = DMatrix::<f64>::zeros(m, n);
    let mut ls = DMatrix::<f64>::zeros(m, n);
    let mut rmse_hist = Vec::new();

    let mut iter = 1usize;
    let mut rng = rand::thread_rng();

    loop {
        let mut y2 = if matches!(
            projection,
            ProjectionKind::SparseJl
                | ProjectionKind::Srht
                | ProjectionKind::Srft
                | ProjectionKind::BlockSparse
        ) {
            let y1 = match projection {
                ProjectionKind::SparseJl => sketch_sparse_jl(&l, rank, &mut rng),
                ProjectionKind::Srht => sketch_srht(&l, rank, &mut rng),
                ProjectionKind::Srft => sketch_srft(&l, rank, &mut rng),
                ProjectionKind::BlockSparse => sketch_block_sparse(&l, rank, &mut rng),
                _ => unreachable!("handled by matches above"),
            };
            let mut y2 = gemm(&l, BlasTranspose::Ordinary, &y1, BlasTranspose::None);
            for _ in 1..iterated_power {
                let y1 = gemm(&l, BlasTranspose::None, &y2, BlasTranspose::None);
                y2 = gemm(&l, BlasTranspose::Ordinary, &y1, BlasTranspose::None);
            }
            y2
        } else {
            let mut y2 = generate_projection(n, rank, projection, &mut rng);
            for _ in 0..iterated_power {
                let y1 = gemm(&l, BlasTranspose::None, &y2, BlasTranspose::None);
                y2 = gemm(&l, BlasTranspose::Ordinary, &y1, BlasTranspose::None);
            }
            y2
        };

        let q = y2.qr().q();
        let l_new = gemm(&l, BlasTranspose::None, &q, BlasTranspose::None);
        let l_new = gemm(&l_new, BlasTranspose::None, &q, BlasTranspose::Ordinary);

        let t = &l - &l_new + &s;
        l = l_new;

        let len = t.len();
        let mut s_data = vec![0.0f64; len];
        if card >= len {
            s_data.copy_from_slice(t.as_slice());
        } else if card > 0 {
            let t_slice = t.as_slice();
            let mut idxs: Vec<usize> = (0..len).collect();
            idxs.select_nth_unstable_by(card - 1, |&a, &b| {
                t_slice[b]
                    .abs()
                    .partial_cmp(&t_slice[a].abs())
                    .unwrap()
            });
            for &idx in &idxs[..card] {
                s_data[idx] = t_slice[idx];
            }
        }
        s = DMatrix::from_column_slice(m, n, &s_data);

        ls = &l + &s;
        let err = rmse(&x, &ls);
        rmse_hist.push(err);
        println!("iter: {} error: {}", iter, err);

        if err <= tol || iter >= max_iter {
            break;
        }
        iter += 1;
    }

    (l, s, ls, rmse_hist)
}

#[pyfunction(signature = (x, rank = 1, card = None, iterated_power = 1, max_iter = 100, tol = 0.001, projection = "gaussian"))]
fn godec(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    rank: usize,
    card: Option<usize>,
    iterated_power: usize,
    max_iter: usize,
    tol: f64,
    projection: &str,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let x_mat = dmatrix_from_pyarray(x)?;
    let total = x_mat.len();
    let card = card.unwrap_or(total);
    let proj = ProjectionKind::from_str(projection).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "projection must be one of: gaussian, rademacher, achlioptas, sparse_jl, srht, srft, block_sparse",
        )
    })?;

    let (l, s, ls, rmse_hist) =
        godec_core(x_mat, rank, card, iterated_power, max_iter, tol, proj);

    let l_py = dmatrix_to_pyarray(py, &l);
    let s_py = dmatrix_to_pyarray(py, &s);
    let ls_py = dmatrix_to_pyarray(py, &ls);
    let rmse_py = PyArray1::from_vec(py, rmse_hist).to_owned();

    Ok((l_py, s_py, ls_py, rmse_py))
}

#[pymodule]
fn godec_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(godec, m)?)?;
    Ok(())
}
