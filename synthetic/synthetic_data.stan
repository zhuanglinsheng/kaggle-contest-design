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

parameters {
	real<lower=1e-8> c_i;
	real<lower=1e-8> c_j;
	real<lower=1e-8> sigma;
	real<lower=1e-8> lambda;
	real<lower=1e-8> r;
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
	vector[T-1] delta_hat_y;
	delta_hat_y[1] = hat_y[1];
	for (t in 2:T-1) {
		delta_hat_y[t] = hat_y[t] - hat_y[t-1];
	}
}

model {
	c_i ~ normal(0, 1);
	c_j ~ normal(0, 1);
	sigma ~ normal(0, 1);
	lambda ~ normal(0, 1);
	r ~ normal(0, 1);

	target += sum(log(r * m_i[hat_t_i_loc])) - r * sum(m_i) * Delta2f;
	target += sum(log(r * m_j[hat_t_j_loc])) - r * sum(m_j) * Delta2f;
	target += normal_lpdf(delta_hat_y |
			effort_gap * Delta2f,
			sqrt((pow(sigma, 2) + 1 / lambda) * Delta2f));
}
