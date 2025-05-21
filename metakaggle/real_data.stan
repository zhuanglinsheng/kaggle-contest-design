
functions {

	/* merge */

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

	/* Ryvkin's model */

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

	vector fn_efforts(
			real y,
			real t,      // current day
			real T,      // deadline day
			real theta,
			real sigma,  // daily innovation shock
			real c_i,    // daily cost
			real c_j     // daily cost
	) {
		if (T <= t) {
			reject("fn_efforts(...): t < T; found (t, T) = ", T, t);
		}
		real sigma_power_2 = pow(sigma, 2);
		real w_i = theta / (sigma_power_2 * c_i);
		real w_j = theta / (sigma_power_2 * c_j);
		real rho_i = (exp(w_i) + exp(-w_j) - 2) / (exp(w_i) - exp(-w_j));
		real rho_j = (exp(w_j) + exp(-w_i) - 2) / (exp(w_j) - exp(-w_i));
		real gamma_rho_i = fn_gamma(rho_i);
		real gamma_rho_j = fn_gamma(rho_j);
		real y_stderr = sigma * sqrt(T - t);
		real density_y = exp(normal_lpdf(y | 0, y_stderr));  // normal density
		real z = y / y_stderr;
		real rho_z = fn_rho(z, gamma_rho_i, gamma_rho_j);
		real K = sigma_power_2 / 2 * (gamma_rho_i + gamma_rho_j) * (1 - pow(rho_z, 2));
		real m_i = density_y * K * (1 + rho_z);
		real m_j = density_y * K * (1 - rho_z);
		vector[2] efforts;
		efforts[1] = m_i;
		efforts[2] = m_j;
		return efforts;  // daily effort rate
	}

}


data {
	///*
	real<lower=0> theta;
	real<lower=0> ratio;
	real<lower=0> Delta2f;
	//*/
	int<lower=0> N_Delta;
	/*
	vector[N_Delta] efforts_i;
	vector[N_Delta] efforts_j;
	*/

	// submission times
	int<lower=0> Ni;  // number of submissions of player i
	int<lower=0> Nj;  // number of submissions of player j
	vector<lower=0, upper=N_Delta>[Ni] hat_t_i;  // submission times of player i
	vector<lower=0, upper=N_Delta>[Nj] hat_t_j;  // submission times of player j

	// real-time leaderboard
	vector[N_Delta] hat_y;  // starts at `t = 1`, ends at `t = deadline`
}

transformed data {
	// for submissions (player i)
	array[Ni] int<lower=1, upper=N_Delta> hat_t_i_timeidx;
	for (ii in 1 : Ni) {
		hat_t_i_timeidx[ii] = to_int(ceil(hat_t_i[ii])) + 1;
	}
	// for submissions (player j)
	array[Nj] int<lower=1, upper=N_Delta> hat_t_j_timeidx;
	for (jj in 1 : Nj) {
		hat_t_j_timeidx[jj] = to_int(ceil(hat_t_j[jj])) + 1;
	}

	// for leaderboard
	///*
	// merge submission times
	array[Ni + Nj] int<lower=1, upper=N_Delta> events_idx =
					merge_ascending_arrays(hat_t_i_timeidx, hat_t_j_timeidx);
	matrix [Ni + Nj, Ni + Nj] unit_cov_y = rep_matrix(0, Ni + Nj, Ni + Nj);
	for (i in 1 : Ni + Nj) {
		for (j in 1 : Ni + Nj) {
			real t_i = events_idx[i] - 1;
			real t_j = events_idx[j] - 1;
			unit_cov_y[i, j] = fmin(t_i, t_j) * Delta2f;
		}
	}
	matrix [Ni + Nj, Ni + Nj] unit_cov_obs = rep_matrix(0, Ni + Nj, Ni + Nj);
	for (i in 1 : Ni + Nj) {
		unit_cov_obs[i, i] = 1.0;
	}
	// */

	// for debug
	//real<lower=5e-1, upper=5>     c_i = 1.2;
	//real<lower=5e-1, upper=5>     c_j = 1.5;
	//real<lower=5e-1, upper=10>    sigma = 2.0;
	//real<lower=1e-6, upper=10>    lambda = 0.5;
	//real<lower=1e-6, upper=1000>  r = 15.0;
}

parameters {
	real<lower=-20, upper=20>     mu_0;
	real<lower=5e-1, upper=5>     c_i;
	real<lower=5e-1, upper=5>     c_j;
	real<lower=5e-1, upper=10>    sigma;
	real<lower=1e-6, upper=10>    lambda;
	//real<lower=1e-6, upper=1000>  r;
}

transformed parameters {
	// calculate m_i, m_j and tilde_y
	///*
	vector<lower=0>[N_Delta] m_i;  // starts at t = 0, ends at t = deadline
	vector<lower=0>[N_Delta] m_j;  // starts at t = 0, ends at t = deadline
	vector[N_Delta + 1] tilde_y;   // starts at t = 0, ends at t = deadline
	tilde_y[1] = mu_0;

	for (i in 1 : N_Delta) {
		vector[2] ms = fn_efforts(  // get `daily` effort rate
				tilde_y[i],
				(i - 1) * Delta2f,  // time = i - 1, transform to float
				N_Delta * Delta2f,  // deadline = T, transform to float
				theta, sigma, c_i, c_j
		);
		m_i[i] = ms[1];
		m_j[i] = ms[2];
		real kalman_gain = sqrt(lambda) * sigma * (hat_y[i] - tilde_y[i]);
		tilde_y[i + 1] = tilde_y[i] + (ms[1] - ms[2] + kalman_gain) * Delta2f;
	}
	//*/

	// intensity (player i)
	vector<lower=0>[N_Delta] intensity_i = ratio * m_i / 24.0;
	vector<lower=0>[Ni] intensity_i_at_events = intensity_i[hat_t_i_timeidx];
	for (idx in 1 : Ni) {
		if (intensity_i_at_events[idx] < 1e-12)
			intensity_i_at_events[idx] = 1e-12;
	}

	// intensity (player j)
	vector<lower=0>[N_Delta] intensity_j = ratio * m_j / 24.0;
	vector<lower=0>[Nj] intensity_j_at_events = intensity_j[hat_t_j_timeidx];
	for (idx in 1 : Nj) {
		if (intensity_j_at_events[idx] < 1e-12)
			intensity_j_at_events[idx] = 1e-12;
	}

	// hat_y: mean and variance
	vector[N_Delta] effort_gap = m_i - m_j;
	vector[Ni + Nj] hat_y_mean;
	for (ii in 1 : Ni + Nj) {
		hat_y_mean[ii] = sum(effort_gap[:events_idx[ii]]) * Delta2f;
	}

	matrix[Ni + Nj, Ni + Nj] hat_y_cov;
	hat_y_cov = pow(sigma, 2) * unit_cov_y + unit_cov_obs / (Delta2f * lambda);
}

model {
	/* priors */
	mu_0 ~ normal(0.0, 5);
	c_i ~ normal(1.0, 5);        // truncated normal
	c_j ~ normal(1.0, 5);        // truncated normal
	sigma ~ normal(1.0, 5);      // truncated normal
	lambda ~ normal(1.0, 5);     // truncated normal
	//r ~ normal(10, 5);

	/* likelihood */
	if (Ni > 1) {
		target += sum(log(intensity_i_at_events)) - sum(intensity_i);
	}
	if (Nj > 1) {
		target += sum(log(intensity_j_at_events)) - sum(intensity_j);
	}

	target += multi_normal_lpdf(hat_y[events_idx] | hat_y_mean, hat_y_cov);

}
