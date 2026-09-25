
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
	int <lower=1> N_tests;
	real<lower=0> theta;
	real<lower=0> ratio;
	real<lower=0> Delta2f;
	int<lower=0> N_Delta;

	array[N_tests] int<lower=1> Ni_arr;
	array[N_tests] int<lower=1> Nj_arr;
	int<lower=1> Ni_max;
	int<lower=1> Nj_max;
	array[N_tests] vector<lower=0, upper=N_Delta>[Ni_max] hat_t_i_arr;
	array[N_tests] vector<lower=0, upper=N_Delta>[Nj_max] hat_t_j_arr;
	array[N_tests] vector[N_Delta + 1] hat_y_arr;
}

transformed data {

	array[N_tests * Ni_max] int hat_t_i_int_arr;
	array[N_tests * Nj_max] int hat_t_j_int_arr;
	array[N_tests] int N_obs_arr;
	array[N_tests * (Ni_max + Nj_max)] int obs_idx_arr =
			rep_array(1, N_tests * (Ni_max + Nj_max));
	array[N_tests] vector[Ni_max + Nj_max] obs_h_arr;
	array[N_tests] matrix[Ni_max + Nj_max, Ni_max + Nj_max] unit_cov_y_arr;

	for (idx_test in 1 : N_tests) {
		int Ni = Ni_arr[idx_test];
		int Nj = Nj_arr[idx_test];

		vector[Ni_max] hat_t_i = hat_t_i_arr[idx_test];
		vector[Nj_max] hat_t_j = hat_t_j_arr[idx_test];

		int base_idx_i = (idx_test - 1) * Ni_max;
		int base_idx_j = (idx_test - 1) * Nj_max;
		int base_idx_ij = (idx_test - 1) * (Ni_max + Nj_max);

		for (ii in 1 : Ni) {
			hat_t_i_int_arr[base_idx_i + ii] = to_int(ceil(hat_t_i[ii]));
		}
		for (jj in 1 : Nj) {
			hat_t_j_int_arr[base_idx_j + jj] = to_int(ceil(hat_t_j[jj]));
		}

		array[Ni + Nj] int events_idx = merge_ascending_arrays(
						hat_t_i_int_arr[base_idx_i + 1 : base_idx_i + Ni],
						hat_t_j_int_arr[base_idx_j + 1 : base_idx_j + Nj]);
		int N_obs = 0;
		array[Ni_max + Nj_max] int obs_idx_full = rep_array(1, Ni_max + Nj_max);
		vector[Ni_max + Nj_max] obs_h_full = rep_vector(1.0, Ni_max + Nj_max);
		for (kk in 1 : Ni + Nj) {
			if (kk == 1 || events_idx[kk] != events_idx[kk - 1]) {
				int previous_step = N_obs == 0 ? 0 : obs_idx_full[N_obs] - 1;
				N_obs += 1;
				obs_idx_full[N_obs] = events_idx[kk] + 1;
				obs_h_full[N_obs] = (events_idx[kk] - previous_step) * Delta2f;
			}
		}
		N_obs_arr[idx_test] = N_obs;
		obs_idx_arr[base_idx_ij + 1 : base_idx_ij + N_obs] = obs_idx_full[1 : N_obs];
		obs_h_arr[idx_test] = obs_h_full;

		matrix[Ni_max + Nj_max, Ni_max + Nj_max] unit_cov_y_full =
				rep_matrix(0, Ni_max + Nj_max, Ni_max + Nj_max);
		for (i in 1 : N_obs) {
			for (j in 1 : N_obs) {
				real t_i = obs_idx_full[i] - 1;
				real t_j = obs_idx_full[j] - 1;
				unit_cov_y_full[i, j] = fmin(t_i, t_j) * Delta2f;
			}
		}
		unit_cov_y_arr[idx_test] = unit_cov_y_full;
	}
	//real<lower=-20, upper=20>     mu_0 = 0.0;
	real<lower=1e-6, upper=1000>  r = ratio;
}

parameters {
	real<lower=1e-1, upper=5>     c_i;
	real<lower=1e-1, upper=5>     c_j;
	real<lower=5e-1, upper=10>    sigma;
	real<lower=1e-6, upper=100>   lambda;
	//real<lower=1e-6, upper=1000>  r;
	real<lower=-20, upper=20>     mu_0;
}

transformed parameters {

	real my_target = 0;

	for (idx_test in 1 : N_tests) {

		/* base
		 */
		int Ni = Ni_arr[idx_test];
		int Nj = Nj_arr[idx_test];

		int base_idx_i = (idx_test - 1) * Ni_max;
		int base_idx_j = (idx_test - 1) * Nj_max;
		int base_idx_ij = (idx_test - 1) * (Ni_max + Nj_max);

		array[Ni] int hat_t_i_int = hat_t_i_int_arr[base_idx_i + 1 : base_idx_i + Ni];
		array[Nj] int hat_t_j_int = hat_t_j_int_arr[base_idx_j + 1 : base_idx_j + Nj];
		int N_obs = N_obs_arr[idx_test];
		array[N_obs] int obs_idx = obs_idx_arr[base_idx_ij + 1 : base_idx_ij + N_obs];
		vector[N_obs] obs_h = obs_h_arr[idx_test][1 : N_obs];

		matrix[Ni_max + Nj_max, Ni_max + Nj_max] unit_cov_y_full = unit_cov_y_arr[idx_test];
		matrix[N_obs, N_obs] unit_cov_y = unit_cov_y_full[1 : N_obs, 1 : N_obs];
		vector[N_Delta + 1] hat_y = hat_y_arr[idx_test];

		/* for each test, generate tilde_y and efforts
		 */
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
		vector[N_Delta] intensity_i = r * m_i / 24.0;
		vector[N_Delta] intensity_j = r * m_j / 24.0;

		/* intensity at events
		 */
		vector[Ni] intensity_i_at_events = intensity_i[hat_t_i_int];
		vector[Nj] intensity_j_at_events = intensity_j[hat_t_j_int];
		my_target += sum(log(intensity_i_at_events)) - sum(intensity_i);
		my_target += sum(log(intensity_j_at_events)) - sum(intensity_j);

		/* hat_y: mean and variance
		 */
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
		my_target += multi_normal_lpdf(hat_y[obs_idx] | hat_y_mean, hat_y_cov);
	}
}

model {
	/* priors */
	c_i ~ lognormal(log(1.0), 0.75);
	c_j ~ lognormal(log(1.0), 0.75);
	sigma ~ lognormal(log(2.0), 0.5);
	lambda ~ lognormal(log(1.0), 0.75);
	//r ~ normal(15, 5);
	mu_0 ~ normal(0.0, 1);       // mean = hat{y}_0, variance is smaller (informative)

	target += my_target;
}
