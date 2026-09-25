functions {
  array[] int merge_ascending_arrays(array[] int arr1, array[] int arr2) {
    int size1 = num_elements(arr1);
    int size2 = num_elements(arr2);
    int i = 1;
    int j = 1;
    int k = 1;
    array[size1 + size2] int merged = rep_array(0, size1 + size2);
    while (i <= size1 && j <= size2) {
      if (arr1[i] < arr2[j]) {
        merged[k] = arr1[i];
        i += 1;
      } else {
        merged[k] = arr2[j];
        j += 1;
      }
      k += 1;
    }
    while (i <= size1) {
      merged[k] = arr1[i];
      i += 1;
      k += 1;
    }
    while (j <= size2) {
      merged[k] = arr2[j];
      j += 1;
      k += 1;
    }
    return merged;
  }

  vector fn_efforts(real y, real t, real T, real theta, real sigma,
                    real c_i, real c_j, real lambda) {
    if (T <= t) {
      reject("fn_efforts requires t < T; found (t, T) = ", t, ", ", T);
    }
    real bar_S = sigma / sqrt(lambda);
    real Q_t = square(sigma) * (T - t) + bar_S;
    real kernel = exp(-square(y) / (2 * Q_t)) / sqrt(2 * pi() * Q_t);
    vector[2] efforts;
    efforts[1] = theta * kernel / c_i;
    efforts[2] = theta * kernel / c_j;
    return efforts;
  }
}

data {
  real<lower=0> theta;
  real<lower=0> Delta2f;
  int<lower=1> N_Delta;
  int<lower=0> Ni;
  int<lower=0> Nj;
  vector<lower=0, upper=N_Delta>[Ni] hat_t_i;
  vector<lower=0, upper=N_Delta>[Nj] hat_t_j;
  vector[N_Delta] hat_y;
}

transformed data {
  array[Ni] int<lower=1, upper=N_Delta> hat_t_i_timeidx;
  array[Nj] int<lower=1, upper=N_Delta> hat_t_j_timeidx;
  array[Ni + Nj] int events_idx;
  int N_obs = 0;
  array[Ni + Nj] int obs_idx_full = rep_array(1, Ni + Nj);
  vector[Ni + Nj] obs_h_full = rep_vector(Delta2f, Ni + Nj);
  matrix[Ni + Nj, Ni + Nj] unit_cov_y_full = rep_matrix(0, Ni + Nj, Ni + Nj);

  for (ii in 1 : Ni) {
    hat_t_i_timeidx[ii] = max(1, to_int(ceil(hat_t_i[ii])));
  }
  for (jj in 1 : Nj) {
    hat_t_j_timeidx[jj] = max(1, to_int(ceil(hat_t_j[jj])));
  }
  events_idx = merge_ascending_arrays(hat_t_i_timeidx, hat_t_j_timeidx);
  for (kk in 1 : Ni + Nj) {
    if (kk == 1 || events_idx[kk] != events_idx[kk - 1]) {
      int previous_idx = N_obs == 0 ? 0 : obs_idx_full[N_obs];
      N_obs += 1;
      obs_idx_full[N_obs] = events_idx[kk];
      obs_h_full[N_obs] = (events_idx[kk] - previous_idx) * Delta2f;
    }
  }
  for (i in 1 : N_obs) {
    for (j in 1 : N_obs) {
      unit_cov_y_full[i, j] = fmin(obs_idx_full[i], obs_idx_full[j]) * Delta2f;
    }
  }
}

parameters {
  // Positive primitives retain only a small numerical lower bound. Their
  // proper priors, rather than arbitrary upper truncation, provide the soft
  // regularization used in estimation.
  real<lower=0.01> c_i;
  real<lower=0.01> c_j;
  real<lower=0.01> sigma;
  real<lower=0.01> lambda;
  real mu_0;
  real<lower=log(0.01)> log_r;
}

model {
  array[N_obs] int obs_idx = obs_idx_full[1 : N_obs];
  vector[N_obs] obs_h = obs_h_full[1 : N_obs];
  matrix[N_obs, N_obs] unit_cov_y = unit_cov_y_full[1 : N_obs, 1 : N_obs];
  vector[N_Delta] intensity_i;
  vector[N_Delta] intensity_j;
  vector[Ni] intensity_i_at_events;
  vector[Nj] intensity_j_at_events;
  vector[N_Delta] effort_gap;
  vector[N_obs] hat_y_mean;
  vector[N_obs] obs_var;
  matrix[N_obs, N_obs] hat_y_cov;
  real bar_S = sigma / sqrt(lambda);
  real r = exp(log_r);
  vector[N_Delta] m_i;
  vector[N_Delta] m_j;
  vector[N_Delta + 1] tilde_y;

  c_i ~ lognormal(log(1.0), 0.75);
  c_j ~ lognormal(log(1.0), 0.75);
  sigma ~ lognormal(log(2.0), 0.5);
  lambda ~ lognormal(log(1.0), 0.75);
  mu_0 ~ normal(hat_y[1], 1.0);
  log_r ~ normal(log(10.0), 0.5);

  tilde_y[1] = mu_0;
  for (i in 1 : N_Delta) {
    vector[2] ms = fn_efforts(
        tilde_y[i], (i - 1) * Delta2f, N_Delta * Delta2f,
        theta, sigma, c_i, c_j, lambda);
    real kalman_weight = -expm1(-sqrt(lambda) * sigma * Delta2f);
    m_i[i] = ms[1];
    m_j[i] = ms[2];
    tilde_y[i + 1] = tilde_y[i] + (ms[1] - ms[2]) * Delta2f
        + kalman_weight * (hat_y[i] - tilde_y[i]);
  }
  intensity_i = r * m_i * Delta2f;
  intensity_j = r * m_j * Delta2f;
  intensity_i_at_events = intensity_i[hat_t_i_timeidx];
  intensity_j_at_events = intensity_j[hat_t_j_timeidx];
  effort_gap = m_i - m_j;

  for (ii in 1 : N_obs) {
    hat_y_mean[ii] = mu_0 + sum(effort_gap[1 : obs_idx[ii]]) * Delta2f;
    obs_var[ii] = 1 / (lambda * obs_h[ii]) + 1e-9;
  }
  hat_y_cov = rep_matrix(bar_S, N_obs, N_obs)
      + square(sigma) * unit_cov_y
      + diag_matrix(obs_var);

  if (Ni > 0) {
    for (ii in 1 : Ni) {
      target += log(fmax(intensity_i_at_events[ii], 1e-12));
    }
    target += -sum(intensity_i);
  }
  if (Nj > 0) {
    for (jj in 1 : Nj) {
      target += log(fmax(intensity_j_at_events[jj], 1e-12));
    }
    target += -sum(intensity_j);
  }
  target += multi_normal_lpdf(hat_y[obs_idx] | hat_y_mean, hat_y_cov);
}
