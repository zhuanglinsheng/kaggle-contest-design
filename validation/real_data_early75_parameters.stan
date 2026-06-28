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

	real fn_gamma(real u) {
		if ((u < -1) || (u > 1)) {
			reject("fn_gamma(x): -1 < x < 1; found x = ", u);
		}
		if (u == -1) {
			return negative_infinity();
		} else if (u == 1) {
			return positive_infinity();
		} else {
			return u / (1 - pow(u, 2)) + atanh(u);
		}
	}

	real fn_invgamma(real x) {
		return atan(0.856 * x) * 2 / pi();
	}

	real fn_rho(real z, real gamma_rho_i, real gamma_rho_j) {
		real loc = normal_cdf(z | 0, 1) * (gamma_rho_i + gamma_rho_j) - gamma_rho_j;
		return fn_invgamma(loc);
	}

	vector fn_efforts(real y, real t, real T, real theta, real sigma, real c_i, real c_j) {
		real sigma_power_2 = pow(sigma, 2);
		real w_i = theta / (sigma_power_2 * c_i);
		real w_j = theta / (sigma_power_2 * c_j);
		real rho_i = (exp(w_i) + exp(-w_j) - 2) / (exp(w_i) - exp(-w_j));
		real rho_j = (exp(w_j) + exp(-w_i) - 2) / (exp(w_j) - exp(-w_i));
		real gamma_rho_i = fn_gamma(rho_i);
		real gamma_rho_j = fn_gamma(rho_j);
		real y_stderr = sigma * sqrt(T - t);
		real density_y = exp(normal_lpdf(y | 0, y_stderr));
		real z = y / y_stderr;
		real rho_z = fn_rho(z, gamma_rho_i, gamma_rho_j);
		real K = sigma_power_2 / 2 * (gamma_rho_i + gamma_rho_j) * (1 - pow(rho_z, 2));
		vector[2] efforts;
		efforts[1] = density_y * K * (1 + rho_z);
		efforts[2] = density_y * K * (1 - rho_z);
		return efforts;
	}

	vector compute_effort_gap(real mu_0, real c_i, real c_j, real sigma, real lambda,
			real theta, real Delta2f, int N_Delta, vector hat_y) {
		vector[N_Delta] effort_gap;
		real tilde_y = mu_0;
		real T = N_Delta * Delta2f;
		for (h in 1:N_Delta) {
			vector[2] ms = fn_efforts(tilde_y, (h - 1) * Delta2f, T, theta, sigma, c_i, c_j);
			real kalman_gain = sqrt(lambda) * sigma * (hat_y[h] - tilde_y);
			effort_gap[h] = ms[1] - ms[2];
			tilde_y += (effort_gap[h] + kalman_gain) * Delta2f;
		}
		return effort_gap;
	}

	real submission_loglik(array[] int event_idx, int N_Delta, real r, vector effort) {
		real out = 0;
		vector[N_Delta] intensity = r * effort / 24.0;
		for (idx in 1:num_elements(event_idx)) {
			out += log(fmax(intensity[event_idx[idx]], 1e-12));
		}
		out -= sum(intensity);
		return out;
	}

	matrix leaderboard_cov(int N_events, matrix unit_cov_y, matrix unit_cov_obs,
			real Delta2f, real sigma, real lambda) {
		return pow(sigma, 2) * unit_cov_y + unit_cov_obs / (Delta2f * lambda);
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
	int<lower=0> N_events = Ni + Nj;
	array[Ni] int<lower=1, upper=N_Delta> hat_t_i_timeidx;
	array[Nj] int<lower=1, upper=N_Delta> hat_t_j_timeidx;
	array[N_events] int<lower=1, upper=N_Delta> events_idx;
	vector[N_events] hat_y_events;
	matrix[N_events, N_events] unit_cov_y = rep_matrix(0, N_events, N_events);
	matrix[N_events, N_events] unit_cov_obs = rep_matrix(0, N_events, N_events);

	if (Ni > 0) {
		for (ii in 1:Ni) {
			hat_t_i_timeidx[ii] = to_int(ceil(hat_t_i[ii])) + 1;
		}
	}
	if (Nj > 0) {
		for (jj in 1:Nj) {
			hat_t_j_timeidx[jj] = to_int(ceil(hat_t_j[jj])) + 1;
		}
	}
	events_idx = merge_ascending_arrays(hat_t_i_timeidx, hat_t_j_timeidx);
	if (N_events > 0) {
		for (i in 1:N_events) {
			hat_y_events[i] = hat_y[events_idx[i]];
			for (j in 1:N_events) {
				unit_cov_y[i, j] = fmin(events_idx[i] - 1, events_idx[j] - 1) * Delta2f;
			}
			unit_cov_obs[i, i] = 1.0;
		}
	}
}

parameters {
	real mu_0;
	real<lower=5e-1, upper=5> c_i;
	real<lower=5e-1, upper=5> c_j;
	real<lower=5e-1, upper=10> sigma;
	real<lower=1e-6> lambda;
	real<lower=1e-2, upper=100> r;
}

model {
	vector[N_Delta] m_i;
	vector[N_Delta] m_j;
	vector[N_Delta] effort_gap;
	real tilde_y = mu_0;
	real T = N_Delta * Delta2f;
	vector[N_events] hat_y_mean;
	matrix[N_events, N_events] hat_y_cov;

	mu_0 ~ normal(hat_y[1], 5);
	c_i ~ normal(1.0, 5);
	c_j ~ normal(1.0, 5);
	sigma ~ normal(1.0, 5);
	lambda ~ normal(1.0, 5);
	r ~ normal(10, 5);

	for (h in 1:N_Delta) {
		vector[2] ms = fn_efforts(tilde_y, (h - 1) * Delta2f, T, theta, sigma, c_i, c_j);
		real kalman_gain = sqrt(lambda) * sigma * (hat_y[h] - tilde_y);
		m_i[h] = ms[1];
		m_j[h] = ms[2];
		effort_gap[h] = ms[1] - ms[2];
		tilde_y += (effort_gap[h] + kalman_gain) * Delta2f;
	}

	if (Ni > 1) {
		target += submission_loglik(hat_t_i_timeidx, N_Delta, r, m_i);
	}
	if (Nj > 1) {
		target += submission_loglik(hat_t_j_timeidx, N_Delta, r, m_j);
	}

	if (N_events > 1) {
		for (ii in 1:N_events) {
			hat_y_mean[ii] = sum(effort_gap[:events_idx[ii]]) * Delta2f;
		}
		hat_y_cov = leaderboard_cov(N_events, unit_cov_y, unit_cov_obs, Delta2f, sigma, lambda);
		target += multi_normal_lpdf(hat_y_events | hat_y_mean, hat_y_cov);
	}
}
