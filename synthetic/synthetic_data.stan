#include "model_effort.stan"

data {
	real<lower=0> theta;
	int<lower=0> T;
	real<lower=0> Delta2f;
	int<lower=0> Ni;
	int<lower=0> Nj;
	array[Ni] int hat_t_i_loc;
	array[Nj] int hat_t_j_loc;
	//vector[Ni] hat_t_i;
	//vector[Nj] hat_t_j;
	vector[T-1] hat_y;  // t = 2 : T
}

transformed data {
	vector[T-1] delta_hat_y;
	delta_hat_y[1] = hat_y[1];
	for (t in 2:T-1) {
		delta_hat_y[t] = hat_y[t] - hat_y[t-1];
	}

	real<lower=1e-1, upper=5> c_i = 0.15;
	real<lower=1e-1, upper=5> c_j = 0.15;
	real<lower=5e-1, upper=10> sigma = 1;
	real<lower=1e-6, upper=1000> lambda = 1;
}

parameters {

	real<lower=1e-6, upper=1> r;          // 0.0 < r      < 1
}

transformed parameters {
	vector<lower=0>[T] m_i;
	vector<lower=0>[T] m_j;
	vector[T] tilde_y;
	tilde_y[1] = 0;
	for (t in 1:T-1) {
		vector[2] ms = fn_efforts(
				tilde_y[t],
				t * Delta2f,  // t is index of hour, transform to day
				T * Delta2f,  // T is index of hour, transform to day
				theta, sigma, c_i, c_j
		);
		m_i[t] = ms[1];
		m_j[t] = ms[2];
		real kalman_gain = sqrt(lambda) * sigma * (hat_y[t] - tilde_y[t]);
		tilde_y[t+1] = tilde_y[t] + (ms[1] - ms[2] + kalman_gain) * Delta2f;
	}
	m_i[T] = 0;
	m_j[T] = 0;
	vector[T-1] effort_gap = m_i[1:T-1] - m_j[1:T-1];
	vector[T-1] intensity_i = m_i[1:T-1] * r;
	vector[T-1] intensity_j = m_j[1:T-1] * r;
}

model {
	/* priors */
	//c_i ~ normal(0.1, 1);     // truncated normal
	//c_j ~ normal(0.1, 1);     // truncated normal
	//sigma ~ normal(0.5, 1);   // truncated normal
	//lambda ~ normal(0.1, 1);  // truncated normal
	r ~ normal(0.1, 1);       // truncated normal

	/* likelihood */
	target += sum(log(intensity_i[hat_t_i_loc])) - sum(intensity_i) * Delta2f;
	//target += sum(log(intensity_j[hat_t_j_loc])) - sum(intensity_j) * Delta2f;
	target += normal_lpdf(delta_hat_y |
			effort_gap * Delta2f, sqrt((pow(sigma, 2) + 1 / lambda) * Delta2f));
}
