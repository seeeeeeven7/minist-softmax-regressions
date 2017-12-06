from mnist import MNIST
import random
import math
import numpy as np

# Parameters
attenuation = 1
epoch_limit = 1000
BATCH_SIZE = 100
step_size = 0.5

# Size
N = 28 * 28 # Size of per test-point
M = 10 # Kinds of labels

class Variable:
	def __init__(self, value = None):
		if value is not None:
			self.value = value
			self.gradient = np.zeros_like(value, dtype=np.float)
		else:
			self.gradient = None
	def __str__(self):
		return '{ value =\n' + str(self.value) + ',\n gradient =\n' + str(self.gradient) + '}';
	def __repr__(self):
		return self.__str__();
	def getOutput(self):
		return self
	def takeInput(self, value):
		self.value = value;
		self.gradient = np.zeros_like(value, dtype=np.float)
	def applyGradient(self, step_size):
		self.value = self.value * attenuation + self.gradient * step_size
		self.gradient = np.zeros_like(self.gradient, dtype=np.float)
	@staticmethod
	def random():
		return Variable(random.random() * 2 - 1);

class Cell:
	def getOutput(self):
		return self.output

class SCell(Cell):
	def __init__(self, input):
		self.input = input

class DCell(Cell):
	def __init__(self, input0, input1):
		self.input0 = input0
		self.input1 = input1

class AddCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(self.input0.getOutput().value + self.input1.getOutput().value)
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient
		self.input1.getOutput().gradient += self.output.gradient

class MatMulCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(np.dot(self.input0.getOutput().value, self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += np.dot(self.output.gradient, self.input1.getOutput().value.T)
		self.input1.getOutput().gradient += np.dot(self.input0.getOutput().value.T, self.output.gradient)

class SoftmaxCell(SCell):
	def forwardPropagation(self):
		self.output = Variable(np.exp(self.input.getOutput().value) / np.sum(np.exp(self.input.getOutput().value)))
	def backwardPropagation(self):
		self.input.getOutput().gradient += -np.sum(self.output.gradient * self.output.value) * self.output.value + self.output.gradient * self.output.value;

class CrossEntropyCell(DCell):
	def forwardPropagation(self):
		self.output = Variable(-np.sum(np.log(self.input0.getOutput().value) * self.input1.getOutput().value))
	def backwardPropagation(self):
		self.input0.getOutput().gradient += self.output.gradient * -self.input1.getOutput().value / self.input0.getOutput().value
		self.input1.getOutput().gradient += self.output.gradient * -np.log(self.input0.getOutput().value)

class Network:
	def __init__(self):
		self.variables = [];
		self.cells = [];
	def getVariablesAmount(self):
		return len(self.variables);
	def getCellsAmount(self):
		return len(self.cells);
	def appendVariable(self, variable):
		self.variables.append(variable);
	def appendCell(self, cell):
		self.cells.append(cell);
	def forwardPropagation(self):
		for cell in self.cells:
			cell.forwardPropagation();
	def backwardPropagation(self):
		for cell in reversed(self.cells):
			# print(cell.getOutput())
			cell.backwardPropagation();
	def applyGradient(self, step_size):
		for variable in self.variables:
			variable.applyGradient(step_size);

# Load tests
mndata = MNIST('data')
images, labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()

# Network
network = Network();

# Variables
W = Variable(np.zeros([N, M]))
network.appendVariable(W);

B = Variable(np.zeros([1, M]))
network.appendVariable(B);

# Inputs Layer
X = Variable(np.zeros([1, N]))
Y = Variable(np.zeros([1, M]))

# Weighted-Sum
matmulCell = MatMulCell(X, W) # X * W => [10, 1]
network.appendCell(matmulCell)

# Bias
addCell = AddCell(matmulCell, B) # X * W + B => [10, 1]
network.appendCell(addCell)

# Softmax
softmaxCell = SoftmaxCell(addCell) # Softmax(X * W + B) => [10, 1]
network.appendCell(softmaxCell)

# Loss (Cross-entropy)
loss = CrossEntropyCell(softmaxCell, Y) # CrossEntropy(Softmax(X * W + B), Y) => Loss
network.appendCell(loss)

# Training
for epoch_index in range(epoch_limit):
	
	# select a batch
	batch_xs = [];
	batch_ys = [];
	for batch_index in range(BATCH_SIZE):
		j = random.randint(0, len(images) - 1)
		x = np.array(images[j])[np.newaxis];
		x = x / 255
		y = labels[j];
		y_ = np.zeros([1, M]);
		y_[0, y] = 1;
		batch_xs.append(x);
		batch_ys.append(y_);

	# train use batch
	batch_loss = 0
	for batch_index in range(BATCH_SIZE):
		x = batch_xs[batch_index];
		y = batch_ys[batch_index];
		X.takeInput(x);
		Y.takeInput(y);
		network.forwardPropagation() # Calculate loss
		batch_loss += loss.getOutput().value
		loss.getOutput().gradient = -1 / BATCH_SIZE
		network.backwardPropagation() # Calculate gradient
	network.applyGradient(step_size)

	# test
	print(epoch_index, '/', epoch_limit, 'loss =', round(batch_loss, 5))
	if epoch_index % 100 == 99:
		precision = 0
		for index in range(len(images_test)):
			x = np.array(images_test[index])[np.newaxis];
			x = x / 255
			y = labels_test[index];
			y_ = np.zeros([1, M]);
			y_[0, y] = 1;
			X.takeInput(x);
			Y.takeInput(y_);
			network.forwardPropagation()
			predict = np.argmax(softmaxCell.getOutput().value)
			if predict == y:
				precision += 1 / len(images_test)
		print('precision =', precision)