
functions {
	/* Ryvkin's model */

	real fn_gamma(real u) {
		if ((u < -1) || (u > 1)) {
			reject("fn_gamma(x): -1 < x < 1; found x = ", u);
		}
		if (u == -1) {
			return negative_infinity();
		} else if (u == 1) {
			return positive_infinity();
		} else {
			return u / (1 - pow(u, 2)) + atanh(u);
		}
	}

	real fn_invgamma(real x) {
		return atan(0.856 * x) * 2 / pi();
	}

	real fn_rho(real z, real gamma_rho_i, real gamma_rho_j) {
		real loc = normal_cdf(z | 0, 1) * (gamma_rho_i + gamma_rho_j) - gamma_rho_j;
		return fn_invgamma(loc);
	}

	vector fn_efforts(
			real y,
			real t,      // current day
			real T,      // deadline day
			real theta,
			real sigma,  // daily innovation shock
			real c_i,    // daily cost
			real c_j     // daily cost
	) {
		if (T <= t) {
			reject("fn_efforts(...): t < T; found (t, T) = ", T, t);
		}
		real sigma_power_2 = pow(sigma, 2);
		real w_i = theta / (sigma_power_2 * c_i);
		real w_j = theta / (sigma_power_2 * c_j);
		real rho_i = (exp(w_i) + exp(-w_j) - 2) / (exp(w_i) - exp(-w_j));
		real rho_j = (exp(w_j) + exp(-w_i) - 2) / (exp(w_j) - exp(-w_i));
		real gamma_rho_i = fn_gamma(rho_i);
		real gamma_rho_j = fn_gamma(rho_j);
		real y_stderr = sigma * sqrt(T - t);
		real density_y = exp(normal_lpdf(y | 0, y_stderr));  // normal density
		real z = y / y_stderr;
		real rho_z = fn_rho(z, gamma_rho_i, gamma_rho_j);
		real K = sigma_power_2 / 2 * (gamma_rho_i + gamma_rho_j) * (1 - pow(rho_z, 2));
		real m_i = density_y * K * (1 + rho_z);
		real m_j = density_y * K * (1 - rho_z);
		vector[2] efforts;
		efforts[1] = m_i;
		efforts[2] = m_j;
		return efforts;  // daily effort rate
	}
}
