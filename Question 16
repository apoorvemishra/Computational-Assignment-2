#include <stdio.h>
#include <math.h>

double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

double euler_method(double t, double y, double h) {
    return y + h * (y - pow(t, 2) + 1);
}

double error_bound(double t, double ti, double h) {
    return h * (0.25 * exp(2) - 1) * (exp(t - ti) - 1);
}

int main() {
    double t = 0.0; // Initial time
    double y = 0.5; // Initial value of y
    double h = 0.2; // Step size

    printf("t\t\tApproximate\t\tExact\t\tError\t\tError Bound\n");
    printf("---------------------------------------------------------------\n");

    while (t <= 2.0) {
        double approximate = euler_method(t, y, h);
        double exact = exact_solution(t);
        double error = fabs(exact - approximate);
        double bound = error_bound(t, 0, h); // Initial time ti is 0
        
        printf("%.2f\t\t%.6f\t%.6f\t%.6f\t%.6f\n", t, approximate, exact, error, bound);
        
        t += h; // Increment time by step size
        y = approximate; // Update y for the next iteration
    }

    return 0;
}
