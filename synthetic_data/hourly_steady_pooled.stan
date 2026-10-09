// Independent contests share six parameters, not latent states or initial draws.
data {
  int<lower=1> J;
  int<lower=1> N;
  real<lower=0> dt;
  real<lower=0> theta;
  real<lower=0> ratio;
  matrix[N+1,J] public_gap;
  array[N,J] int<lower=0> count_i;
  array[N,J] int<lower=0> count_j;
  real<lower=0> mu_prior_sd;
}
transformed data {
  matrix[N,J] counts;
  array[N] int any_readout;
  int total_i = 0;
  int total_j = 0;
  for (n in 1:N) {
    any_readout[n] = 0;
    for (j in 1:J) {
      counts[n,j] = count_i[n,j]+count_j[n,j];
      if (counts[n,j]>0) any_readout[n] = 1;
      total_i += count_i[n,j];
      total_j += count_j[n,j];
    }
  }
}
parameters {
  real<lower=0.1,upper=5> c_i;
  real<lower=0.1,upper=5> c_j;
  real<lower=0.5,upper=10> sigma;
  real<lower=1e-6,upper=100> lambda;
  real<lower=-20,upper=20> mu_0;
  real log_r;
}
transformed parameters {
  real r = exp(log_r);
  real S0 = sigma/sqrt(lambda);
  real log_likelihood;
  {
    vector[J] perceived = rep_vector(mu_0,J);
    vector[J] latent_mean = rep_vector(mu_0,J);
    vector[J] latent_var = rep_vector(S0,J);
    vector[J] elapsed = rep_vector(0,J);
    real sig2 = square(sigma);
    real gain = -expm1(-sigma*sqrt(lambda)*dt);
    real cost_sum = inv(c_i)+inv(c_j);
    real cost_diff = inv(c_i)-inv(c_j);
    log_likelihood = total_i*(log_r-log(c_i)-log(24))
                   + total_j*(log_r-log(c_j)-log(24));
    for (n in 1:N) {
      real Q = S0+sig2*((N-n+1)*dt);
      vector[J] log_kernel = log(theta)-0.5*log(2*pi()*Q)-square(perceived)/(2*Q);
      vector[J] kernel = exp(log_kernel);
      vector[J] drift = kernel*(cost_diff*dt);
      log_likelihood -= r*cost_sum*dt*sum(kernel);
      if (any_readout[n]) log_likelihood += counts[n]*log_kernel;
      perceived += drift+gain*(public_gap[n]'-perceived);
      latent_mean += drift;
      latent_var += sig2*dt;
      elapsed += dt;
      if (any_readout[n]) {
        for (j in 1:J) {
          if (counts[n,j]>0) {
            real noise = inv(lambda*elapsed[j]);
            real v = latent_var[j]+noise;
            real innovation = public_gap[n+1,j]-latent_mean[j];
            log_likelihood += normal_lpdf(public_gap[n+1,j] | latent_mean[j],sqrt(v));
            latent_mean[j] += latent_var[j]/v*innovation;
            latent_var[j] *= noise/v;
            elapsed[j] = 0;
          }
        }
      }
    }
  }
}
model {
  target += log_likelihood;
  // One common prior per pool, with exactly the Table 1 supports.
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
}
