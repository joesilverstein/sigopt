# Question 2
# Joe Silverstein

n = 30;
x = unifrnd(0,1,n,1);
A = zeros(n);
K = @(x,z) min(x,z) - x*z;
for i = 1:n
	for j = 1:n
		A(i,j) = K(x(i),x(j));
	endfor
endfor

y = x .* (1 - x);
c = inv(A) * y;

B = zeros(n);
K = @(x,z) exp(-(x - z)^2);
for i = 1:n
        for j = 1:n
                B(i,j) = K(x(i),x(j));
        endfor
endfor

[U, S, V] = svd(B);
s = diag(S);
# Use MATLAB's rank cutoff value, based on machine precision:
# http://www.mathworks.com/help/matlab/ref/rank.html
# This is the formula for the numerical rank.
tol = max(size(B)) * eps(max(s));
rank = sum(s > tol);
cond = max(s) / min(s);
# Condition number is very large, so B is ill-conditioned.

mu = 10e-10;
I = eye(n);
d = inv(B + mu * I) * y;

# I'm guessing there are typos in this question. VTilde has to be a square matrix to be invertible, so I'm assuming it meant the upper left 8x8 block of V. Similarly for U. Then I also used the upper left 8x8 block of B to compute the Frobenius norm.
UTilde = U(1:8, 1:8);
VTilde = V(1:8, 1:8);
STilde = S(1:8, 1:8);
BTilde = UTilde * STilde * inv(VTilde);
frob = norm(BTilde - B(1:8,1:8), "fro");

# Typos again? If BTilde is 8x8, then I has to be 8x8, y has to be 8x1, and d has to be 8x1.
dTilde = inv(BTilde + mu * eye(8)) * y(1:8); # Octave warns that matrix is singular
twoNorm = norm(d(1:8) - dTilde, 2);



