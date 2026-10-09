// Original hourly synthetic likelihood, extended only to free S0.
// Sequential Gaussian innovations equal the original dense covariance density.
data {
  int<lower=1> N;
  real<lower=0> dt;
  real<lower=0> theta;
  real<lower=0> ratio;
  int<lower=0,upper=1> estimate_r;
  vector[N+1] public_gap;
  array[N] int<lower=0> count_i;
  array[N] int<lower=0> count_j;
  real<lower=0> mu_prior_sd;
  real S0_prior_logmean;
  real<lower=0> S0_prior_logsd;
}
parameters {
  real<lower=0.1,upper=5> c_i;
  real<lower=0.1,upper=5> c_j;
  real<lower=0.5,upper=10> sigma;
  real<lower=1e-6,upper=100> lambda;
  real<lower=-20,upper=20> mu_0;
  real<lower=0.01> S0;
  vector[estimate_r] log_r_free;
}
transformed parameters {
  real r = estimate_r == 1 ? exp(log_r_free[1]) : ratio;
}
model {
  real perceived = mu_0;
  real latent_mean = mu_0;
  real latent_var = S0;
  real elapsed = 0;
  real steady = sigma / sqrt(lambda);
  real rate = sigma * sqrt(lambda);
  c_i ~ lognormal(log(1.0),0.75);
  c_j ~ lognormal(log(1.0),0.75);
  sigma ~ lognormal(log(2.0),0.5);
  lambda ~ lognormal(log(1.0),0.75);
  mu_0 ~ normal(0,mu_prior_sd);
  S0 ~ lognormal(S0_prior_logmean,S0_prior_logsd);
  if (estimate_r == 1) log_r_free[1] ~ normal(log(ratio),0.35);
  for (n in 1:N) {
    real t = (n-1)*dt;
    real z = tanh(rate*t);
    real S = steady*(S0+steady*z)/(steady+S0*z);
    real Q = S+square(sigma)*(N*dt-t);
    real log_kernel = log(theta)-0.5*log(2*pi()*Q)-square(perceived)/(2*Q);
    real mi = exp(log_kernel)/c_i;
    real mj = exp(log_kernel)/c_j;
    real drift = (mi-mj)*dt;
    // Event times are measured in hours, as in the original Stan model.
    target += count_i[n]*(log(r)+log_kernel-log(c_i)-log(24));
    target += count_j[n]*(log(r)+log_kernel-log(c_j)-log(24));
    target += -r*(mi+mj)*dt;
    perceived += drift-expm1(-lambda*S*dt)*(public_gap[n]-perceived);
    latent_mean += drift;
    latent_var += square(sigma)*dt;
    elapsed += dt;
    if (count_i[n]+count_j[n] > 0) {
      real noise = 1/(lambda*elapsed);
      real v = latent_var+noise;
      real innovation = public_gap[n+1]-latent_mean;
      target += normal_lpdf(public_gap[n+1] | latent_mean,sqrt(v));
      latent_mean += latent_var/v*innovation;
      latent_var *= noise/v;
      elapsed = 0;
    }
  }
}
