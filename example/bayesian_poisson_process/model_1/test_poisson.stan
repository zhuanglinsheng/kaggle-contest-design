
data {
	real<lower=0> T;
	int<lower=0> N;
	array[N] real<lower=0,upper=T> events;
}

parameters {
	real<lower=0, upper=1> lambda;  // hour arrival rate
}

model {
	// prior
	lambda ~ normal(0.5, 1);

	// log likelihood
	target += N * log(lambda) - lambda * T;
}
