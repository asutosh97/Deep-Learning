from numpy import *
# y = mx + c
# m is slope and c is intercept

def compute_error_for_line(m, c, points):
	total_error = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y - m * x - c) ** 2

	return total_error/float(len(points))

def step_gradient(m_current, c_current, points, learning_rate):
	m_gradient = 0
	c_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient += -2/N * x * (y - ((m_current * x) + c_current))
		c_gradient += -2/N * (y - ((m_current * x) + c_current))
	m_new = m_current - (learning_rate * m_gradient)
	c_new = c_current - (learning_rate * c_gradient)
	return [m_new, c_new]


def gradient_descent_runner(points, initial_m, initial_c, learning_rate, number_of_iterations):
	m = initial_m
	c = initial_c
	for i in range(number_of_iterations):
		m, c = step_gradient(m, c, points, learning_rate)
	return [m, c]
def run():
	#getting data from the CSV file
	points = genfromtxt("data.csv", delimiter=",")

	# declaring primary variables to be used in the program
	learning_rate = 0.0001
	initial_m = 0
	initial_c = 0
	number_of_iterations = 5000
	print "Initial error when m = {0}, c = {1}, error = {2}".format(initial_m, initial_c, compute_error_for_line(initial_m, initial_c, points))
	print "Running"
	[m, c] = gradient_descent_runner(points,initial_c, initial_m, learning_rate, number_of_iterations)
	print "Final error when m = {0}, c = {1}, error = {2}".format(m, c, compute_error_for_line(m, c, points))



if __name__ == '__main__':
	run()