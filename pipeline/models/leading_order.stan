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
  vector[6] prior_mean;
  vector<lower=0>[6] prior_sd;
  real<lower=0> sigma_floor;
  real<lower=0> theta;
  real<lower=0> Delta2f;
  int<lower=1> N_Delta;
  int<lower=1, upper=N_Delta> count_start_idx;
  real<lower=0> T_contest;
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

  if (T_contest < N_Delta * Delta2f) {
    reject("The observation window must not exceed the contest deadline");
  }

  for (ii in 1 : Ni) {
    hat_t_i_timeidx[ii] = max(1, to_int(ceil(hat_t_i[ii])));
    if (hat_t_i_timeidx[ii] < count_start_idx) {
      reject("Submission precedes the count observation window");
    }
  }
  for (jj in 1 : Nj) {
    hat_t_j_timeidx[jj] = max(1, to_int(ceil(hat_t_j[jj])));
    if (hat_t_j_timeidx[jj] < count_start_idx) {
      reject("Submission precedes the count observation window");
    }
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
}

parameters {
  // Positive primitives retain only a small numerical lower bound. Their
  // proper priors, rather than arbitrary upper truncation, provide the soft
  // regularization used in estimation.
  real<lower=0.01> c_i;
  real<lower=0.01> c_j;
  real<lower=sigma_floor> sigma;
  real<lower=0.01> lambda;
  real mu_0;
  real<lower=log(0.01)> log_r;
}

model {
  array[N_obs] int obs_idx = obs_idx_full[1 : N_obs];
  vector[N_obs] obs_h = obs_h_full[1 : N_obs];
  vector[N_Delta] intensity_i;
  vector[N_Delta] intensity_j;
  vector[Ni] intensity_i_at_events;
  vector[Nj] intensity_j_at_events;
  vector[N_Delta] effort_gap;
  vector[N_Delta] integrated_drift;
  real bar_S = sigma / sqrt(lambda);
  real r = exp(log_r);
  vector[N_Delta] m_i;
  vector[N_Delta] m_j;
  vector[N_Delta + 1] tilde_y;
  real residual_mean = 0;
  real residual_variance = bar_S;

  c_i ~ lognormal(prior_mean[1], prior_sd[1]);
  c_j ~ lognormal(prior_mean[2], prior_sd[2]);
  sigma ~ lognormal(prior_mean[3], prior_sd[3]);
  lambda ~ lognormal(prior_mean[4], prior_sd[4]);
  mu_0 ~ normal(prior_mean[5], prior_sd[5]);
  log_r ~ normal(prior_mean[6], prior_sd[6]);

  tilde_y[1] = mu_0;
  for (i in 1 : N_Delta) {
    vector[2] ms = fn_efforts(
        tilde_y[i], (i - 1) * Delta2f, T_contest,
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
  integrated_drift = cumulative_sum(effort_gap) * Delta2f;

  // Scalar Gaussian innovations are exactly the dense Brownian covariance
  // likelihood, without allocating/factorizing an N_obs by N_obs matrix.
  for (ii in 1 : N_obs) {
    real prediction_variance = residual_variance + square(sigma) * obs_h[ii];
    real measurement_variance = 1 / (lambda * obs_h[ii]) + 1e-9;
    real innovation_variance = prediction_variance + measurement_variance;
    real innovation = hat_y[obs_idx[ii]] - mu_0 - integrated_drift[obs_idx[ii]] - residual_mean;
    target += normal_lpdf(innovation | 0, sqrt(innovation_variance));
    residual_mean += prediction_variance / innovation_variance * innovation;
    residual_variance = prediction_variance * measurement_variance / innovation_variance;
  }

  if (Ni > 0) {
    for (ii in 1 : Ni) {
      target += log(fmax(intensity_i_at_events[ii], 1e-12));
    }
  }
  if (Nj > 0) {
    for (jj in 1 : Nj) {
      target += log(fmax(intensity_j_at_events[jj], 1e-12));
    }
  }
  // Backfilled hours initialize the recursion but are not zero-count data.
  // The exposure terms also apply to players with no observed submissions.
  target += -sum(intensity_i[count_start_idx : N_Delta]);
  target += -sum(intensity_j[count_start_idx : N_Delta]);
}
