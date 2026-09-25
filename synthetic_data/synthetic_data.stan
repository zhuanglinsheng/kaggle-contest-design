
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

	/* Leading-order equilibrium in Theorem 1 */
	vector fn_efforts(
			real y,
			real t,      // current day
			real T,      // deadline day
			real theta,
			real sigma,  // daily innovation shock
			real c_i,    // daily cost
			real c_j,    // daily cost
			real lambda  // signal precision
	) {
		if (T <= t) {
			reject("fn_efforts(...): t < T; found (t, T) = ", T, t);
		}
		real bar_S = sigma / sqrt(lambda);
		real Q_t = square(sigma) * (T - t) + bar_S;
		real kernel = exp(-square(y) / (2 * Q_t)) / sqrt(2 * pi() * Q_t);
		vector[2] efforts;
		efforts[1] = theta * kernel / c_i;
		efforts[2] = theta * kernel / c_j;
		return efforts;  // daily effort rate
	}

}


data {
	///*
	real<lower=0> theta;
	real<lower=0> ratio;
	real<lower=0> Delta2f;
	int<lower=0, upper=1> estimate_r;
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
	vector[N_Delta + 1] hat_y;  // starts at `t = 0`, ends at `t = deadline`
}

transformed data {
	// for submissions (player i)
	array[Ni] int<lower=1, upper=N_Delta> hat_t_i_timeidx;
	array[Nj] int<lower=1, upper=N_Delta> hat_t_j_timeidx;
	array[Ni + Nj] int events_idx;
	int N_obs = 0;
	array[Ni + Nj] int obs_idx_full = rep_array(1, Ni + Nj);
	vector[Ni + Nj] obs_h_full = rep_vector(1.0, Ni + Nj);
	matrix[Ni + Nj, Ni + Nj] unit_cov_y_full = rep_matrix(0, Ni + Nj, Ni + Nj);
	for (ii in 1 : Ni) {
		hat_t_i_timeidx[ii] = to_int(ceil(hat_t_i[ii]));
	}
	// for submissions (player j)
	for (jj in 1 : Nj) {
		hat_t_j_timeidx[jj] = to_int(ceil(hat_t_j[jj]));
	}

	events_idx = merge_ascending_arrays(hat_t_i_timeidx, hat_t_j_timeidx);
	for (kk in 1 : Ni + Nj) {
		if (kk == 1 || events_idx[kk] != events_idx[kk - 1]) {
			int previous_step = N_obs == 0 ? 0 : obs_idx_full[N_obs] - 1;
			N_obs += 1;
			obs_idx_full[N_obs] = events_idx[kk] + 1;
			obs_h_full[N_obs] = (events_idx[kk] - previous_step) * Delta2f;
		}
	}
	for (i in 1 : N_obs) {
		for (j in 1 : N_obs) {
			real t_i = obs_idx_full[i] - 1;
			real t_j = obs_idx_full[j] - 1;
			unit_cov_y_full[i, j] = fmin(t_i, t_j) * Delta2f;
		}
	}

	// for debug
	//real<lower=5e-1, upper=5>     c_i = 1.2;
	//real<lower=5e-1, upper=5>     c_j = 1.5;
	//real<lower=5e-1, upper=10>    sigma = 2.0;
	//real<lower=1e-6, upper=100>   lambda = 0.5;
	//real<lower=-20, upper=20>     mu_0 = 0.0;
}

parameters {
	real<lower=1e-1, upper=5>     c_i;
	real<lower=1e-1, upper=5>     c_j;
	real<lower=5e-1, upper=10>    sigma;
	real<lower=1e-6, upper=100>   lambda;
	real<lower=-20, upper=20>     mu_0;
	real log_r;
}

model {
	array[N_obs] int obs_idx = obs_idx_full[1 : N_obs];
	vector[N_obs] obs_h = obs_h_full[1 : N_obs];
	matrix[N_obs, N_obs] unit_cov_y = unit_cov_y_full[1 : N_obs, 1 : N_obs];

	/* priors */
	c_i ~ lognormal(log(1.0), 0.75);
	c_j ~ lognormal(log(1.0), 0.75);
	sigma ~ lognormal(log(2.0), 0.5);
	lambda ~ lognormal(log(1.0), 0.75);
	mu_0 ~ normal(0.0, 1);       // mean = hat{y}_0, variance is smaller (informative)
	log_r ~ normal(log(ratio), 0.35);
	real r = estimate_r == 1 ? exp(log_r) : ratio;

	// calculate m_i, m_j and tilde_y
	///*
	vector[N_Delta] m_i;  // starts at t = 0, ends at t = deadline
	vector[N_Delta] m_j;  // starts at t = 0, ends at t = deadline
	vector[N_Delta + 1] tilde_y;   // starts at t = 0, ends at t = deadline
	tilde_y[1] = mu_0;

	for (i in 1 : N_Delta) {
		vector[2] ms = fn_efforts(  // get `daily` effort rate
				tilde_y[i],
				(i - 1) * Delta2f,  // time = i - 1, transform to float
				N_Delta * Delta2f,  // deadline = T, transform to float
				theta, sigma, c_i, c_j, lambda
		);
		m_i[i] = ms[1];
		m_j[i] = ms[2];
		real kalman_weight = -expm1(-sqrt(lambda) * sigma * Delta2f);
		tilde_y[i + 1] = tilde_y[i] + (ms[1] - ms[2]) * Delta2f
				+ kalman_weight * (hat_y[i] - tilde_y[i]);
	}
	//*/

	// intensity (player i)
	vector[N_Delta] intensity_i = r * m_i / 24.0;
	vector[Ni] intensity_i_at_events = intensity_i[hat_t_i_timeidx];

	// intensity (player j)
	vector[N_Delta] intensity_j = r * m_j / 24.0;
	vector[Nj] intensity_j_at_events = intensity_j[hat_t_j_timeidx];

	// hat_y: mean and variance
	vector[N_Delta] effort_gap = m_i - m_j;
	vector[N_obs] hat_y_mean;
	for (ii in 1 : N_obs) {
		hat_y_mean[ii] = mu_0 + sum(effort_gap[:obs_idx[ii] - 1]) * Delta2f;
	}

	matrix[N_obs, N_obs] hat_y_cov;
	vector[N_obs] obs_var;
	for (ii in 1 : N_obs) {
		obs_var[ii] = 1 / (lambda * obs_h[ii]);
	}
	real bar_S = sigma / sqrt(lambda);
	hat_y_cov = rep_matrix(bar_S, N_obs, N_obs)
			+ square(sigma) * unit_cov_y
			+ diag_matrix(obs_var);

	/* likelihood */
	if (Ni > 1) {
		target += sum(log(intensity_i_at_events)) - sum(intensity_i);
	}
	if (Nj > 1) {
		target += sum(log(intensity_j_at_events)) - sum(intensity_j);
	}
	target += multi_normal_lpdf(hat_y[obs_idx] | hat_y_mean, hat_y_cov);
}
