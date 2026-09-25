
functions {

	/* merge */

	array[] int merge_ascending_arrays(array[] int arr1, array[] int arr2) {
		int size1 = num_elements(arr1);
		int size2 = num_elements(arr2);
		int i = 1;
		int j = 1;
		int k = 1;
		array[size1 + size2] int merged = rep_array(0, size1 + size2);

		while (i <= size1 && j <= size2) {
			if (arr1[i] < arr2[j]) {
				merged[k] = arr1[i];
				i += 1;
			} else {
				merged[k] = arr2[j];
				j += 1;
			}
			k += 1;
		}
		while (i <= size1) {
			merged[k] = arr1[i];
			i += 1;
			k += 1;
		}
		while (j <= size2) {
			merged[k] = arr2[j];
			j += 1;
			k += 1;
		}
		return merged;
	}

	/* Leading-order equilibrium in Theorem 1 */
	vector fn_efforts(
			real y,
			real t,      // current day
			real T,      // deadline day
			real theta,
			real sigma,  // daily innovation shock
			real c_i,    // daily cost
			real c_j,    // daily cost
			real lambda  // signal precision
	) {
		if (T <= t) {
			reject("fn_efforts(...): t < T; found (t, T) = ", T, t);
		}
		real bar_S = sigma / sqrt(lambda);
		real Q_t = square(sigma) * (T - t) + bar_S;
		real kernel = exp(-square(y) / (2 * Q_t)) / sqrt(2 * pi() * Q_t);
		vector[2] efforts;
		efforts[1] = theta * kernel / c_i;
		efforts[2] = theta * kernel / c_j;
		return efforts;  // daily effort rate
	}

}
