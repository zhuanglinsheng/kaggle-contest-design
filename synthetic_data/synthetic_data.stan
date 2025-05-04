#include "model_effort.stan"

data {
	///*
	real<lower=0> theta;
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
	///*
	vector[N_Delta + 1] hat_y;  // starts at `t = 0`, ends at `t = deadline`
	//*/
}

transformed data {
	// for submissions (player i)
	array[Ni] int<lower=0, upper=N_Delta - 1> hat_t_i_idx;
	for (ii in 1 : Ni) {
		hat_t_i_idx[ii] = to_int(hat_t_i[ii]) + 1;
	}
	/*
	array[Ni] int<lower=0, upper=N_Delta - 1> hat_t_i_floor_plus1;
	array[Ni] int<lower=1, upper=N_Delta    > hat_t_i_ceil_plus1;
	vector<lower=0, upper=1>[Ni] events_i_ratio;
	for (ii in 1 : Ni) {
		hat_t_i_floor_plus1[ii] = to_int(hat_t_i[ii]) + 1;
		hat_t_i_ceil_plus1[ii] = hat_t_i_floor_plus1[ii] + 1;
		events_i_ratio[ii] = hat_t_i[ii] + 1 - hat_t_i_floor_plus1[ii];
	}
	*/

	// for submissions (player j)
	array[Nj] int<lower=0, upper=N_Delta - 1> hat_t_j_idx;
	for (jj in 1 : Nj) {
		hat_t_j_idx[jj] = to_int(hat_t_j[jj]) + 1;
	}
	/*
	array[Nj] int<lower=0, upper=N_Delta - 1> hat_t_j_floor_plus1;
	array[Nj] int<lower=1, upper=N_Delta    > hat_t_j_ceil_plus1;
	vector<lower=0, upper=1>[Nj] events_j_ratio;
	for (ii in 1 : Nj) {
		hat_t_j_floor_plus1[ii] = to_int(hat_t_j[ii]) + 1;
		hat_t_j_ceil_plus1[ii] = hat_t_j_floor_plus1[ii] + 1;
		events_j_ratio[ii] = hat_t_j[ii] + 1 - hat_t_j_floor_plus1[ii];
	}
	*/

	// for leaderboard
	///*
	vector[N_Delta] delta_hat_y;  // starts at first Delta, ends at last Delta
	for (i in 1 : N_Delta) {
		delta_hat_y[i] = hat_y[i + 1] - hat_y[i];
	}
	//*/
	// for debug
	//real<lower=5e-1, upper=10> sigma = 5.0;
	real<lower=1e-6, upper=1000>  r = 50;
}

parameters {
	real<lower=5e-1, upper=5> c_i;
	real<lower=5e-1, upper=5> c_j;
	real<lower=5e-1, upper=10> sigma;
	real<lower=1e-6, upper=100> lambda;
}

transformed parameters {
	// calculate m_i, m_j and tilde_y
	///*
	vector<lower=0>[N_Delta] m_i;  // starts at t = 0, ends at t = deadline
	vector<lower=0>[N_Delta] m_j;  // starts at t = 0, ends at t = deadline
	vector[N_Delta + 1] tilde_y;       // starts at t = 0, ends at t = deadline
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
	vector<lower=0>[N_Delta] intensity_i = r * m_i / 24.0;
	vector<lower=0>[Ni] intensity_i_at_events = intensity_i[hat_t_i_idx];
	/*
	vector<lower=0>[N_Delta + 1] intensity_i = r * m_i / 24;
	vector<lower=0>[Ni] intensity_i_at_events_floor = intensity_i[hat_t_i_floor_plus1];
	vector<lower=0>[Ni] intensity_i_at_events_ceil = intensity_i[hat_t_i_ceil_plus1];
	vector<lower=0>[Ni] intensity_i_at_events =
		intensity_i_at_events_floor .* (1 - events_i_ratio) +
		intensity_i_at_events_ceil  .* events_i_ratio;
	*/
	// intensity (player j)
	vector<lower=0>[N_Delta] intensity_j = r * m_j / 24.0;
	vector<lower=0>[Nj] intensity_j_at_events = intensity_j[hat_t_j_idx];
}

model {
	/* priors */
	c_i ~ normal(1, 1);     // truncated normal
	c_j ~ normal(1, 1);     // truncated normal
	sigma ~ normal(1, 1);   // truncated normal
	lambda ~ normal(0.5, 1);  // truncated normal
	r ~ normal(10, 100);       // truncated normal

	/* likelihood */
	if (Ni > 1) {
		target += sum(log(intensity_i_at_events)) - sum(intensity_i);
	}
	if (Nj > 1) {
		target += sum(log(intensity_j_at_events)) - sum(intensity_j);
	}
	target += normal_lpdf(delta_hat_y |
			(m_i - m_j) * Delta2f, sqrt((pow(sigma, 2) + 1 / lambda) * Delta2f));
}
