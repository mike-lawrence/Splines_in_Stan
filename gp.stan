functions{
	# GP: computes noiseless Gaussian Process
	vector GP(real volatility, real amplitude, vector normal01, int n, real[] x ) {

		# cov_mat: covariance matrix
		matrix[n,n] cov_mat ;

		# amplitude_sq: square of the amplitude
		real amplitude_sq ;
		amplitude_sq = amplitude^2 ;

		#use built-in function to compute covariance matrix
		cov_mat = cov_exp_quad(x, amplitude, 1/volatility) ;

		#jitter diagonal for positive definiteness
		for(i in 1:n){
			cov_mat[i,i] = amplitude_sq + .001 ;
		}

		# combine with normal01 & return
		return(cholesky_decompose(cov_mat) * normal01 ) ;
	}

}
data {
	# n: num unique x values
	int<lower=1> n ;
	# Y: outcomes
	vector[n] y ;
	# x: predictors
	real x[n] ;
}
parameters {
	# amplitude: GP parameter, scale of latent function f
	real<lower=0> amplitude ;
	# noise: scale of measurement noise of y given f
	real<lower=0> noise ;
	# normal01: dummy variable to speed up computation of GP
	#   (See "Cholesky Factored and Transformed Implementation" from manual)
	vector[n] normal01 ;
	# volatility_helper: dummy variable used to imply cauchy(0,1) prior on volatility
	#   (See "Reparameterizing the Cauchy" from manual)
	real<lower=0,upper=pi()/2> volatility_helper ;
}
transformed parameters{
	# f: latent function values
	vector[n] f ;
	# volatility: GP parameter, x-axis scale of influence between points (wiggliness)
	#       a.k.a. inverse-lengthscale
	real<lower=0> volatility ;
	volatility = tan(volatility_helper) ; #implies volatility ~ cauchy(0,1), peaked @ zero with heavy tail
	# f as GP
	f = GP(volatility, amplitude, normal01, n, x ) ;
}
model {
	# volatility has an implicit cauchy(0,1) prior
	# prior on amplitude
	amplitude ~ weibull(2,1) ; #peaked around .8
	# prior on noise
	noise ~ weibull(2,1) ; #peaked around .8
	# normal01 as standard normal
	normal01 ~ normal(0,1) ;
	# y as f plus noise
	y ~ normal(f,noise) ;
}
