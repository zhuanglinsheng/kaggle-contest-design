#include "model_effort.stan"

data {
	///*
	real<lower=0> theta;
	real<lower=0> ratio;
	real<lower=0> Delta2f;
	//*/
	int<lower=0> N_Delta;
	///*
	vector[N_Delta] efforts_i;
	vector[N_Delta] efforts_j;
	//*/

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
	//*/

	// intensity (player i)
	vector<lower=0>[N_Delta] intensity_i = ratio * m_i / 24.0;
	vector<lower=0>[Ni] intensity_i_at_events = intensity_i[hat_t_i_timeidx];

	// intensity (player j)
	vector<lower=0>[N_Delta] intensity_j = ratio * m_j / 24.0;
	vector<lower=0>[Nj] intensity_j_at_events = intensity_j[hat_t_j_timeidx];

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
