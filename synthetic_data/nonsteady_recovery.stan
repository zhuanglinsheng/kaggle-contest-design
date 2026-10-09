// Chapter 3 approximate likelihood; full nonlinear equilibrium is used ONLY
// by the external data generator. Scalar innovations integrate out y_0.
data {
  int<lower=1> J;
  int<lower=1> N;
  array[J] int<lower=1, upper=N> start;
  array[J] int<lower=1, upper=N> end;
  vector<lower=0>[J] T;
  vector<lower=0>[J] ratio;
  vector<lower=0>[N] dt;
  vector<lower=0>[N] t;
  vector[N] held;
  vector[N] obs;
  array[N] int<lower=0, upper=2> player;
  real<lower=0> theta;
  int<lower=0, upper=1> estimate_r;
  real mu_prior_center;
}
parameters {
  real<lower=0.01> c_i;
  real<lower=0.01> c_j;
  real<lower=0.01> sigma;
  real<lower=0.01> lambda;
  real<lower=0.01> S0;
  real mu_0;
  vector<lower=0.01>[estimate_r] r_free;
}
model {
  real steady = sigma / sqrt(lambda);
  real rate = sigma * sqrt(lambda);
  c_i ~ lognormal(log(1.0), 0.75);
  c_j ~ lognormal(log(1.0), 0.75);
  sigma ~ lognormal(log(2.0), 0.5);
  lambda ~ lognormal(log(1.0), 0.75);
  S0 ~ lognormal(log(2.0), 0.75);
  mu_0 ~ normal(mu_prior_center, 1.0);
  if (estimate_r == 1) r_free ~ lognormal(log(10.0), 0.5);
  for (j in 1:J) {
    real r = estimate_r == 1 ? r_free[1] : ratio[j];
    real perceived = mu_0;
    real latent_mean = mu_0;
    real latent_var = S0;
    real elapsed = 0;
    for (n in start[j]:end[j]) {
      real z = tanh(rate*t[n]);
      real S = steady * (S0 + steady*z) / (steady + S0*z);
      real Q = S + square(sigma)*(T[j]-t[n]);
      real kernel = theta * exp(-square(perceived)/(2*Q)) / sqrt(2*pi()*Q);
      real mi = kernel/c_i;
      real mj = kernel/c_j;
      real gap = mi-mj;
      real gain = lambda*S;
      real weight = -expm1(-gain*dt[n]);
      // Exponential step with effort and gain frozen at interval start.
      // Only past public readouts enter the perceived-state predictor.
      perceived += weight*(held[n]-perceived + gap/gain);
      latent_mean += gap*dt[n];
      latent_var += square(sigma)*dt[n];
      elapsed += dt[n];
      target += -r*(mi+mj)*dt[n];
      if (player[n] > 0) {
        real noise_var = 1/(lambda*elapsed);
        real predictive_var = latent_var+noise_var;
        real innovation = obs[n]-latent_mean;
        // Intensity just BEFORE reading the current public gap.
        real te = t[n]+dt[n];
        real ze = tanh(rate*te);
        real Se = steady*(S0+steady*ze)/(steady+S0*ze);
        real Qe = Se+square(sigma)*(T[j]-te);
        target += log(r*theta) - log(player[n] == 1 ? c_i : c_j)
                  - 0.5*log(2*pi()*Qe) - square(perceived)/(2*Qe);
        target += normal_lpdf(obs[n] | latent_mean, sqrt(predictive_var));
        latent_mean += latent_var/predictive_var*innovation;
        latent_var = latent_var*noise_var/predictive_var;
        elapsed = 0;
      }
    }
  }
}
