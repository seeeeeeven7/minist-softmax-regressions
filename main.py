from mnist import MNIST
import random

mndata = MNIST('data')
images, labels = mndata.load_training()

class AddNeuron:
	def __init__(self, inputNeuron0, inputNeuron1):
		pass

class Variable:
	def __init__(self, value, gradient):
		self.value = value;
		self.gradient = gradient;
	def __str__(self):
		return '(' + str(self.value) + ',' + str(self.gradient) + ')';
	def __repr__(self):
		return self.__str__();
	@staticmethod
	def random():
		return Variable(random.random() * 2 - 1, 0);

# 各类
N = len(images) # 训练数据集中，数据点的数量
M = 10 # 数据点标签的种类数(0~9)

W = [[Variable.random() for i in range(N)] for j in range(M)]
B = []
print(W)