#include "model_effort.stan"

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
	array[N_tests * (Ni_max + Nj_max)] int events_idx_arr;
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
			hat_t_i_int_arr[base_idx_i + ii] = 1 + to_int(hat_t_i[ii]);
		}
		for (jj in 1 : Nj) {
			hat_t_j_int_arr[base_idx_j + jj] = 1 + to_int(hat_t_j[jj]);
		}

		array[Ni + Nj] int events_idx = merge_ascending_arrays(
						hat_t_i_int_arr[base_idx_i + 1 : base_idx_i + Ni],
						hat_t_j_int_arr[base_idx_j + 1 : base_idx_j + Nj]);
		events_idx_arr[base_idx_ij + 1 : base_idx_ij + Ni + Nj] = events_idx;

		//print(">>>>> events_idx = ", events_idx);

		matrix [Ni + Nj, Ni + Nj] unit_cov_y = rep_matrix(0, Ni + Nj, Ni + Nj);
		for (i in 1 : Ni + Nj) {
			for (j in 1 : Ni + Nj) {
				real t_i = events_idx[i] - 1;
				real t_j = events_idx[j] - 1;
				unit_cov_y[i, j] = fmin(t_i, t_j) * Delta2f;
			}
		}
		unit_cov_y_arr[idx_test][1 : Ni + Nj, 1 : Ni + Nj] = unit_cov_y;
	}
	matrix [Ni_max + Nj_max, Ni_max + Nj_max] unit_cov_obs =
			rep_matrix(0, Ni_max + Nj_max, Ni_max + Nj_max);
	for (ii in 1 : Ni_max + Nj_max) {
		unit_cov_obs[ii, ii] = 1.0;
	}
}

parameters {
	real<lower=5e-1, upper=5>     c_i;
	real<lower=5e-1, upper=5>     c_j;
	real<lower=5e-1, upper=10>    sigma;
	real<lower=1e-6, upper=10>    lambda;
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
		array[Ni + Nj] int events_idx = events_idx_arr[base_idx_ij + 1 : base_idx_ij + Ni + Nj];

		matrix[Ni_max + Nj_max, Ni_max + Nj_max] unit_cov_y_full = unit_cov_y_arr[idx_test];
		matrix[Ni + Nj, Ni + Nj] unit_cov_y = unit_cov_y_full[1 : Ni + Nj, 1 : Ni + Nj];
		vector[N_Delta + 1] hat_y = hat_y_arr[idx_test];

		/* for each test, generate tilde_y and efforts
		 */
		vector[N_Delta] m_i;  // starts at t = 0, ends at t = deadline
		vector[N_Delta] m_j;  // starts at t = 0, ends at t = deadline
		vector[N_Delta + 1] tilde_y;   // starts at t = 0, ends at t = deadline
		tilde_y[1] = 0;
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
		vector[N_Delta] intensity_i = ratio * m_i / 24.0;
		vector[N_Delta] intensity_j = ratio * m_j / 24.0;

		/* intensity at events
		 */
		vector[Ni] intensity_i_at_events = intensity_i[hat_t_i_int];
		vector[Nj] intensity_j_at_events = intensity_j[hat_t_j_int];
		my_target += sum(log(intensity_i_at_events)) - sum(intensity_i);
		my_target += sum(log(intensity_j_at_events)) - sum(intensity_j);

		/* hat_y: mean and variance
		 */
		vector[N_Delta] effort_gap = m_i - m_j;
		vector[Ni + Nj] hat_y_mean;
		for (ii in 1 : Ni + Nj) {
			hat_y_mean[ii] = sum(effort_gap[:events_idx[ii]]) * Delta2f;
		}
		matrix[Ni + Nj, Ni + Nj] hat_y_cov;
		matrix[Ni + Nj, Ni + Nj] eyes = unit_cov_obs[1 : Ni + Nj, 1 : Ni + Nj];
		// print(eyes);
		print(unit_cov_y);
		hat_y_cov = pow(sigma, 2) * unit_cov_y + eyes / (Delta2f * lambda);
		my_target += multi_normal_lpdf(hat_y[events_idx] | hat_y_mean, hat_y_cov);
	}
}

model {
	/* priors */
	c_i ~ normal(1.0, 5);        // truncated normal
	c_j ~ normal(1.0, 5);        // truncated normal
	sigma ~ normal(1.0, 5);      // truncated normal
	lambda ~ normal(1.0, 5);     // truncated normal

	target += my_target;
}
