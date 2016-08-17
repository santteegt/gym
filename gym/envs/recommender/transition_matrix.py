import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.cross_validation import KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
#from sklearn.utils.extmath import np.dot

import scipy.spatial.distance as ssd
from scipy.stats import spearmanr as spearman
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
import time
import os

warnings.simplefilter("error")

users = 6040
items = 3952


class Util(object):

	@staticmethod
	def adjusted_cosine(X, Y, N):
		"""
		Developed by: https://gist.github.com/irgmedeiros/5859643

		Considering the rows of X (and Y=X) as vectors, compute the
		distance matrix between each pair of vectors after normalize or adjust
		the vector using the N vector. N vector contains the mean of the values
		of each feature vector from X and Y.

		This correlation implementation is equivalent to the cosine similarity
		since the data it receives is assumed to be centered -- mean is 0. The
		correlation may be interpreted as the cosine of the angle between the two
		vectors defined by the users' preference values.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples_1, n_features]

		Y : {array-like, sparse matrix}, shape = [n_samples_2, n_features]

		N: {array-like, sparse matrix}, shape = [n_samples_3, n_features]

		Returns
		-------
		distances : {array, sparse matrix}, shape = [n_samples_1, n_samples_2]

		"""

		# X, Y = check_pairwise_arrays(X, Y)
		# TODO: fix next line
		# N, _ = check_pairwise_arrays(N, None)

		# should not need X_norm_squared because if you could precompute that as
		# well as Y, then you should just pre-compute the output and not even
		# call this function.

		# TODO: Fix to work with sparse matrices.
		if issparse(X) or issparse(Y) or issparse(N):
			raise ValueError('Adjusted cosine does not yet support sparse matrices.')

		if X is Y:
			X = Y = np.asanyarray(X)
		else:
			X = np.asanyarray(X)
			Y = np.asanyarray(Y)

		if X.shape[1] != Y.shape[1] != N.shape[1]:
			raise ValueError("Incompatible dimension for X, Y and N matrices")

		X = X - N
		Y = Y - N

		XY = 1 - ssd.cdist(X, Y, 'cosine')

		return XY

class TransitionProbability(object):

	def __init__(self, shape, train=True, raw_data=None, alpha=0.1, beta=0.9, data_dir='data'):
		self.__shape = shape
		self.__raw_data = raw_data
		self.rating_matrix_filename = os.path.join(data_dir, "rmatrix.npy")
		self.similarity_matrix_filename = os.path.join(data_dir, "simmatrix.npy")
		self.transition_matrix_filename = os.path.join(data_dir, "trmatrix.npy")
		self.ranking_matrix_filename = os.path.join(data_dir, "rankmatrix.npy")
		if train:
			assert raw_data is not None, "Ratings data must be provided for training the model"
			self.__ratings_matrix, self.__sim_matrix = self._get_initial_matrices()
			self._crossValidation()
			self.__tr_matrix = self.get_transition_matrix(generate=True, beta=beta)
			self.__ranking_matrix = self._compute_rankings(alpha=0.1, scale=True)
			np.save(self.rating_matrix_filename, self.__ratings_matrix, allow_pickle=False)
			np.save(self.similarity_matrix_filename, self.__sim_matrix, allow_pickle=False)
			np.save(self.transition_matrix_filename, self.__tr_matrix, allow_pickle=False)
			np.save(self.ranking_matrix_filename, self.__ranking_matrix, allow_pickle=False)
			# np.savetxt(self.ranking_matrix_filename+'.txt', self.__ranking_matrix)
		else:
			# self.__ratings_matrix = np.load(self.rating_matrix_filename, "r", allow_pickle=False)
			# self.__sim_matrix = np.load(self.similarity_matrix_filename, "r", allow_pickle=False)
			self.__tr_matrix = np.load(self.transition_matrix_filename, "r", allow_pickle=False)
			self.__ranking_matrix = np.load(self.ranking_matrix_filename, "r", allow_pickle=False)
			# print "Rating matrix shape: {} Similarity matrix shape: {}".format(self.__ratings_matrix.shape, self.__sim_matrix.shape)

		print "Transition probability matrix shape: {} Ranking matrix shape: {}".format(self.__tr_matrix.shape,
																		   self.__ranking_matrix.shape)

	# def similarity_item(data):
	# 	print "Hello"
	# 	#f_i_d = open("sim_item_based.txt","w")
	# 	item_similarity_cosine = np.zeros((items,items))
	# 	# item_similarity_jaccard = np.zeros((items,items))
	# 	# item_similarity_pearson = np.zeros((items,items))
	# 	for item1 in range(items):
	# 		print item1
	# 		for item2 in range(items):
	# 			# if np.count_nonzero(data[:,item1]) and np.count_nonzero(data[:,item2]):
	# 				# item_similarity_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[:,item1],data[:,item2])
	# 			print "i dim {} j dim {}".format(data[:,item1].shape, data[:,item2].shape)
	# 			item_similarity_cosine[item1][item2] = 1-scipy.spatial.distance.cdist(data[:,item1], data[:,item2], metric='cosine')
	# 				# item_similarity_jaccard[item1][item2] = 1-scipy.spatial.distance.jaccard(data[:,item1],data[:,item2])
	# 				# try:
	# 				# 	if not math.isnan(scipy.stats.pearsonr(data[:,item1],data[:,item2])[0]):
	# 				# 		item_similarity_pearson[item1][item2] = scipy.stats.pearsonr(data[:,item1],data[:,item2])[0]
	# 				# 	else:
	# 				# 		item_similarity_pearson[item1][item2] = 0
	# 				# except:
	# 				# 	item_similarity_pearson[item1][item2] = 0
    #
	# 			#f_i_d.write(str(item1) + "," + str(item2) + "," + str(item_similarity_cosine[item1][item2]) + "," + str(item_similarity_jaccard[item1][item2]) + "," + str(item_similarity_pearson[item1][item2]) + "\n")
	# 	#f_i_d.close()
	# 	return item_similarity_cosine#, item_similarity_jaccard, item_similarity_pearson

	def _get_initial_matrices(self):
		'''

		Returns:

		'''

		Mat = np.zeros(self.__shape)

		# Ratings matrix
		for e in self.__raw_data:
			Mat[e[0] - 1][e[1] - 1] = e[2]

		# Mat = Mat[100:110,:10] + 1
		# print Mat
		rows, cols = Mat.shape
		# nonzero_counts = [np.count_nonzero(Mat[row]) for row in range(rows)]
		sum_user_ratings = np.sum(Mat, axis=1)
		# mean_user_ratings = sum_user_ratings / nonzero_counts
		mean_user_ratings = sum_user_ratings / cols
		# for col in range(cols):
		# 	Mat[:,col] = Mat[:,col] - (((Mat[:,col] > 0) * 1) * mean_user_ratings)

		# print Mat
		# return [], []

		print "Initializing Similarity matrix process"
		start_time = time.time()
		# sim_item_cosine, sim_item_jaccard, sim_item_pearson = similarity_item(Mat)
		# sim_item_cosine = similarity_item(Mat)
		print "Rating matrix properties: {} ".format(Mat.shape)
		# transpose for compute the similarity of items
		transp = np.matrix.transpose(Mat)
		# with np.errstate(divide='ignore',invalid='ignore'):
		# 	sim_item_cosine = 1 - scipy.spatial.distance.cdist(transp, transp, metric='cosine')
		sim_item_cosine = Util.adjusted_cosine(transp, transp, np.matrix.transpose(mean_user_ratings))

		# sim_item_cosine, sim_item_jaccard, sim_item_pearson = np.random.rand(items,items), np.random.rand(items,items), np.random.rand(items,items)

		print "Process ends in {} seconds".format(time.time() - start_time)
		print "Sim matrix properties: {}".format(sim_item_cosine.shape)

		return Mat, sim_item_cosine


	def _crossValidation(self, n_folds=5):
		'''

		Returns:

		'''
		start_time = time.time()
		print "Initializing cross validation process"
		# k_fold = KFold(n=len(data), n_folds=10)
		k_fold = KFold(n=len(self.__raw_data), n_folds=n_folds)

		rmse_cosine = []

		for train_indices, test_indices in k_fold:
			train = [self.__raw_data[i] for i in train_indices]
			test = [self.__raw_data[i] for i in test_indices]

			M = np.zeros(self.__shape)

			for e in train:
				M[e[0]-1][e[1]-1] = e[2]

			true_rate = []
			pred_rate_cosine = []

			for e in test:
				user = e[0]
				item = e[1]
				true_rate.append(e[2])

				pred_cosine = 3.0

				#item-based
				if np.count_nonzero(M[:,item-1]):
					sim_cosine = self.__sim_matrix[item-1]
					ind = (M[user-1] > 0)
					normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
					if normal_cosine > 0:
						pred_cosine = np.dot(sim_cosine,M[user-1])/normal_cosine

				# clip predition under the range [0, 5]
				pred_cosine = np.clip(pred_cosine, 0, 5)

				# print str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine)# + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson)
				pred_rate_cosine.append(pred_cosine)

			rmse_cosine.append(sqrt(mean_squared_error(true_rate, pred_rate_cosine)))

			# print str(sqrt(mean_squared_error(true_rate, pred_rate_cosine))) # + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))

		rmse_cosine = sum(rmse_cosine) / float(len(rmse_cosine))

		print "Cross-validation process finished in {} seconds".format(time.time() - start_time)
		print "Final RMSE value with {}-fold cross validation: {}".format(n_folds, rmse_cosine) # + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson)

		f_rmse = open("rmse_item.txt","w")
		f_rmse.write(str(rmse_cosine))# + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

		rmse = [rmse_cosine]#, rmse_jaccard, rmse_pearson]
		req_sim = rmse.index(min(rmse))

		# print req_sim
		f_rmse.write(str(req_sim))
		f_rmse.close()

	def get_transition_matrix(self, generate=False, beta=0.9):
		if generate:
			sum_Sim = np.sum(self.__sim_matrix)
			self.__tr_matrix = beta * self.__sim_matrix / sum_Sim + (1. - beta) / self.__shape[1]

		return self.__tr_matrix

	def get_transitions_per_item(self, item_id):
		'''
		Gets the transition probability for next items given the current item
		Args:
		    item_id:

		Returns:

		'''

		return self.get_transition_matrix()[item_id - 1]

	def _compute_rankings(self, alpha=0.1, scale=False):
		'''


		Args:
		    alpha: probability of continuing the random walk. The lower the alpha value is, better predictions when
		    random walk is based on the adjusted cosine similarity

		Returns:

		'''
		random_walk_length = np.ones_like(self.__tr_matrix) - alpha * self.__tr_matrix
		P_hat = alpha * self.__tr_matrix * np.linalg.pinv(random_walk_length)
		self.__ranking_matrix = np.dot(self.__ratings_matrix, P_hat)

		return np.clip(self._scale_rows(self.__ranking_matrix) if scale else self.__ranking_matrix, 0.0, 5.0)


	def _scale_rows(self, matrix):
		max_indexes = np.argmax(matrix, axis=1)
		scales = [ 5. / matrix[row][max_indexes[row]] for row in np.arange(len(max_indexes)) ]
		scales = np.expand_dims(scales, axis=1)
		return scales * matrix

	def get_rankings_per_user(self, user_id):
		'''
		Gets the predicted ratings per user
		Args:
		    user_id:

		Returns:

		'''
		return self.__ranking_matrix[user_id - 1]


	def predictRating(self, toBeRated):

		#f = open("toBeRated.csv","r")

		pred_rate = []

		#fw = open('result2.csv','w')
		fw_w = open('result2.csv','w')

		l = len(toBeRated["user"])
		for e in range(l):
			user = toBeRated["user"][e]
			item = toBeRated["item"][e]

			pred = 3.0

			#item-based
			if np.count_nonzero(self.__ratings_matrix[:,item-1]):
				sim = self.__sim_matrix[item-1]
				ind = (self.__ratings_matrix[user-1] > 0)
				print "ind -> {}".format(ind)
				#ind[item-1] = False
				normal = np.sum(np.absolute(sim[ind]))
				if normal > 0:
					pred = np.dot(sim,self.__ratings_matrix[user-1])/normal

			if pred < 0:
				pred = 0

			if pred > 5:
				pred = 5

			pred_rate.append(pred)
			print str(user) + "," + str(item) + "," + str(pred)
			#fw.write(str(user) + "," + str(item) + "," + str(pred) + "\n")
			fw_w.write(str(pred) + "\n")

		#fw.close()
		fw_w.close()


def readingFile(filename, split=','):
	f = open(filename, "r")
	data = []
	for row in f:
		r = row.split(split)
		e = [int(r[0]), int(r[1]), int(r[2])]
		data.append(e)
	return data

if __name__ == "__main__":

	recommend_data = readingFile(sys.argv[1], split="\t")
	inst = TransitionProbability(train=False, raw_data=recommend_data, shape=(943, 1682))
	#crossValidation(recommend_data)
	if len(sys.argv) > 2:
		f = open(sys.argv[2], "r")
		toBeRated = {"user": [], "item": []}
		for row in f:
			r = row.split(',')
			toBeRated["item"].append(int(r[1]))
			toBeRated["user"].append(int(r[0]))

		f.close()
		inst.predictRating(toBeRated)

