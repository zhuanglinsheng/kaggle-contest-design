// Same hourly observation likelihood as hourly_nonsteady.stan.
// All prior constants are retained for marginal-likelihood computation.
functions {
  real hourly_loglik(int N, real dt, real theta, vector public_gap,
                    array[] int count_i, array[] int count_j,
                    real c_i, real c_j, real sigma, real lambda,
                    real mu_0, real S0, real log_r) {
    real perceived = mu_0;
    real latent_mean = mu_0;
    real latent_var = S0;
    real elapsed = 0;
    real steady = sigma / sqrt(lambda);
    real rate = sigma * sqrt(lambda);
    real ans = 0;
    for (n in 1:N) {
      real t = (n-1)*dt;
      real z = tanh(rate*t);
      real S = steady*(S0+steady*z)/(steady+S0*z);
      real Q = S+square(sigma)*(N*dt-t);
      real log_kernel = log(theta)-0.5*log(2*pi()*Q)-square(perceived)/(2*Q);
      real mi = exp(log_kernel)/c_i;
      real mj = exp(log_kernel)/c_j;
      real drift = (mi-mj)*dt;
      ans += count_i[n]*(log_r+log_kernel-log(c_i)-log(24));
      ans += count_j[n]*(log_r+log_kernel-log(c_j)-log(24));
      ans -= exp(log_r)*(mi+mj)*dt;
      perceived += drift-expm1(-lambda*S*dt)*(public_gap[n]-perceived);
      latent_mean += drift;
      latent_var += square(sigma)*dt;
      elapsed += dt;
      if (count_i[n]+count_j[n] > 0) {
        real noise = 1/(lambda*elapsed);
        real v = latent_var+noise;
        real innovation = public_gap[n+1]-latent_mean;
        ans += normal_lpdf(public_gap[n+1] | latent_mean,sqrt(v));
        latent_mean += latent_var/v*innovation;
        latent_var *= noise/v;
        elapsed = 0;
      }
    }
    return ans;
  }
}
data {
  int<lower=1> N;
  real<lower=0> dt;
  real<lower=0> theta;
  real<lower=0> ratio;
  vector[N+1] public_gap;
  array[N] int<lower=0> count_i;
  array[N] int<lower=0> count_j;
  int<lower=0,upper=1> free_S0;
  int<lower=0,upper=1> use_prior;
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
  real log_r;
  vector<lower=0.01>[free_S0] S0_free;
}
transformed parameters {
  real S0 = free_S0 == 1 ? S0_free[1] : sigma/sqrt(lambda);
  real r = exp(log_r);
  real log_likelihood = hourly_loglik(N,dt,theta,public_gap,count_i,count_j,
                                    c_i,c_j,sigma,lambda,mu_0,S0,log_r);
}
model {
  target += log_likelihood;
  if (use_prior == 1) {
    target += lognormal_lpdf(c_i | 0,0.75)
      - log_diff_exp(lognormal_lcdf(5 | 0,0.75),lognormal_lcdf(0.1 | 0,0.75));
    target += lognormal_lpdf(c_j | 0,0.75)
      - log_diff_exp(lognormal_lcdf(5 | 0,0.75),lognormal_lcdf(0.1 | 0,0.75));
    target += lognormal_lpdf(sigma | log(2),0.5)
      - log_diff_exp(lognormal_lcdf(10 | log(2),0.5),lognormal_lcdf(0.5 | log(2),0.5));
    target += lognormal_lpdf(lambda | 0,0.75)
      - log_diff_exp(lognormal_lcdf(100 | 0,0.75),lognormal_lcdf(1e-6 | 0,0.75));
    target += normal_lpdf(mu_0 | 0,mu_prior_sd)
      - log_diff_exp(normal_lcdf(20 | 0,mu_prior_sd),normal_lcdf(-20 | 0,mu_prior_sd));
    target += normal_lpdf(log_r | log(ratio),0.35);
    if (free_S0 == 1)
      target += lognormal_lpdf(S0 | S0_prior_logmean,S0_prior_logsd)
        - lognormal_lccdf(0.01 | S0_prior_logmean,S0_prior_logsd);
  }
}
