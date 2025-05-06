//#include "merge_array.stan"
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
	vector[N_Delta + 1] hat_y;  // starts at `t = 0`, ends at `t = deadline`
}

transformed data {
	// for submissions (player i)
	array[Ni] int<lower=1, upper=N_Delta> hat_t_i_timeidx;
	for (ii in 1 : Ni) {
		hat_t_i_timeidx[ii] = to_int(ceil(hat_t_i[ii])) + 1;
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
	array[Nj] int<lower=1, upper=N_Delta> hat_t_j_timeidx;
	for (jj in 1 : Nj) {
		hat_t_j_timeidx[jj] = to_int(ceil(hat_t_j[jj])) + 1;
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
	// merge submission times
	array[Ni + Nj] int<lower=1, upper=N_Delta> events_idx = merge_ascending_arrays(hat_t_i_timeidx, hat_t_j_timeidx);
	vector[Ni + Nj] delta_hat_y;
	delta_hat_y[1] = hat_y[1 + events_idx[1]] - hat_y[1];
	// hat_y[1] = \hat{y}_0 = 0
	for (ii in 2 : Ni + Nj) {
		delta_hat_y[ii] = hat_y[events_idx[ii]] - hat_y[events_idx[ii] - 1];
	}
	//print("events_idx  = ", events_idx);
	//print("delta_hat_y = ", delta_hat_y);
	// */

	// for debug
	//real<lower=5e-1, upper=10> sigma = 5.0;

}

parameters {
	real<lower=5e-1, upper=5>     c_i;
	real<lower=5e-1, upper=5>     c_j;
	real<lower=5e-1, upper=10>    sigma;
	real<lower=1e-6, upper=10>    lambda;
	real<lower=1e-6, upper=1000>  r;
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
	vector<lower=0>[N_Delta] intensity_i = r * m_i / 24.0;
	vector<lower=0>[Ni] intensity_i_at_events = intensity_i[hat_t_i_timeidx];
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
	vector<lower=0>[Nj] intensity_j_at_events = intensity_j[hat_t_j_timeidx];

	// delta_hat_y: mean and variance
	vector[N_Delta] effort_gap = m_i - m_j;
	vector[Ni + Nj] delta_hat_y_mean;
	vector[Ni + Nj] delta_hat_y_duration;
	delta_hat_y_mean[1] = sum(effort_gap[:events_idx[1]]) * Delta2f;
	delta_hat_y_duration[1] = events_idx[1] * Delta2f;

	for (i in 2 : Ni + Nj) {
		int time1 = events_idx[i - 1];
		int time2 = events_idx[i];
		delta_hat_y_mean[i] = sum(effort_gap[time1 : time2]) * Delta2f;
		delta_hat_y_duration[i] = (time2 - time1) * Delta2f;
		if (delta_hat_y_duration[i] < 0) {
			print("Error: Element ", i, "--", delta_hat_y_duration[i], " is negative.");
			reject("Vector contains negative elements");
		}
		else if (delta_hat_y_duration[i] < 1e-6) {
			delta_hat_y_duration[i] = 1e-6;
		}
	}
	vector[Ni + Nj] delta_hat_y_std = sqrt(delta_hat_y_duration * pow(sigma, 2) + 2/(Delta2f * lambda));
}

model {
	/* priors */
	c_i ~ normal(1, 1);        // truncated normal
	c_j ~ normal(1, 1);        // truncated normal
	sigma ~ normal(0.5, 1);    // truncated normal
	lambda ~ normal(0.5, 1);   // truncated normal
	r ~ normal(10, 200);       // truncated normal

	/* likelihood */
	if (Ni > 1) {
		target += sum(log(intensity_i_at_events)) - sum(intensity_i);
	}
	if (Nj > 1) {
		target += sum(log(intensity_j_at_events)) - sum(intensity_j);
	}

	print("len = ", num_elements(delta_hat_y), delta_hat_y);
	print("len = ", num_elements(delta_hat_y_mean), delta_hat_y_mean);
	print("len = ", num_elements(delta_hat_y_duration), delta_hat_y_duration);

	target += normal_lpdf(delta_hat_y | delta_hat_y_mean, delta_hat_y_std);
}
