
data {
	int<lower=0> hours;                 // number of hours
	vector<lower=0>[hours] effort;  // per hour daily effort
	int<lower=0> N;
	vector<lower=0, upper=hours>[N] events;
}

transformed data {
	array[N] int<lower=0, upper=hours-1> events_floor;
	array[N] int<lower=0, upper=hours-1> events_ceil;
	vector<lower=0, upper=1>[N] events_ratio;

	for (ii in 1:N) {
		events_floor[ii] = 1 + to_int(events[ii]);  // 1 index
		events_ceil[ii] = events_floor[ii] + 1;
		events_ratio[ii] = 1 + events[ii] - events_floor[ii];
	}
}

parameters {
	real<lower=1e-2, upper=10> r;  // ratio: intensity to effort
}

transformed parameters {
	vector<lower=0>[hours] intensity = r * effort / 24.0;
	vector<lower=0>[N] intensity_at_events_floor = intensity[events_floor];
	vector<lower=0>[N] intensity_at_events_ceil = intensity[events_ceil];
	vector<lower=0>[N] intensity_at_events =
		intensity_at_events_floor .* (1 - events_ratio) +
		intensity_at_events_ceil  .* events_ratio;
}

model {
	// prior
	r ~ normal(1, 10);

	// log likelihood
	target += sum(log(intensity_at_events)) - sum(intensity);
}
