import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from math import sqrt
import warnings
import sys

import scipy.spatial.distance as ssd
from scipy.sparse import issparse
from scipy.sparse import csr_matrix, csc_matrix
import time
import os

warnings.simplefilter("error")


class Util(object):
    @staticmethod
    def adjusted_cosine(X, Y, N):
        """
        DEPRECATED. NOW PROCESS IS USING SKLEARN METHOD FOR SPARSE MATRICES
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
        self.items_collection_filename = os.path.join(data_dir, "items_collection.npy")
        self.rating_matrix_filename = os.path.join(data_dir, "rmatrix.npy")
        self.similarity_matrix_filename = os.path.join(data_dir, "simmatrix.npy")
        self.transition_matrix_filename = os.path.join(data_dir, "trmatrix.npy")
        self.ranking_matrix_filename = os.path.join(data_dir, "rankmatrix.npy")
        if train:
            assert raw_data is not None, "Ratings data must be provided for training the model"

            # ACTUAL COMPUTATION OF TRANSITION MATRIX

            self.__ratings_matrix, self.__sim_matrix = self._get_initial_matrices()
            # np.savetxt(self.similarity_matrix_filename + '.txt', self.__sim_matrix)
            # self._crossValidation()
            self.__tr_matrix = self.get_transition_matrix(generate=True, beta=beta)

            # USE TRANSTITION MATRIX TO COMPUTE RANKING MATRIX
            # self.__ranking_matrix = self._compute_rankings(alpha=alpha, scale=True)
            self.__ranking_matrix = self._compute_rankings(alpha=alpha, scale=False)

            items_collection = self.__ratings_matrix.T.toarray()
            np.save(self.items_collection_filename, items_collection, allow_pickle=False)
            self.__ratings_matrix = self.__ratings_matrix.toarray()
            np.save(self.rating_matrix_filename, self.__ratings_matrix, allow_pickle=False)
            np.save(self.similarity_matrix_filename, self.__sim_matrix, allow_pickle=False)
            np.save(self.transition_matrix_filename, self.__tr_matrix, allow_pickle=False)
            np.save(self.ranking_matrix_filename, self.__ranking_matrix, allow_pickle=False)
            # np.savetxt(self.ranking_matrix_filename+'.txt', self.__ranking_matrix)
        else: # LOAD MODEL
            self.__ratings_matrix = np.load(self.rating_matrix_filename, "r", allow_pickle=False)
            self.__sim_matrix = np.load(self.similarity_matrix_filename, "r", allow_pickle=False)
            self.__tr_matrix = np.load(self.transition_matrix_filename, "r", allow_pickle=False)
            self.__ranking_matrix = np.load(self.ranking_matrix_filename, "r", allow_pickle=False)

        print "Transition probability matrix shape: {} Ranking matrix shape: {}".format(self.__tr_matrix.shape,
                                                                                        self.__ranking_matrix.shape)

    def _get_initial_matrices(self):
        """
        Generates user rating matrix and item similarity matrix
        Returns:

        """

        Mat = np.zeros(self.__shape)

        # Ratings matrix
        for e in self.__raw_data:
            Mat[e[0] - 1][e[1] - 1] = e[2]

        sum_rows = np.sum(Mat, axis=0)

        Mat = csr_matrix(Mat)



        # Mat = Mat[100:110,:10] + 1
        # print Mat
        rows, cols = Mat.shape
        # nonzero_counts = [np.count_nonzero(Mat[row]) for row in range(rows)]

        # sum_user_ratings = np.sum(Mat, axis=1) # commented
        # sum_user_ratings = Mat.sum(axis=1)

        # mean_user_ratings = sum_user_ratings / nonzero_counts

        nonzero_counts = np.sum(Mat.toarray() > 0, axis=1)

        # mean_user_ratings = sum_user_ratings / cols
        mean_user_ratings = np.sum(Mat.toarray(), axis=1) / nonzero_counts

        print "Initializing Similarity matrix process"
        start_time = time.time()
        print "Rating matrix properties: {} ".format(Mat.shape)

        # transpose for compute the similarity of items
        Mat_transpose = Mat.transpose()
        mur_transpose = mean_user_ratings.transpose()
        X = Mat_transpose# - mur_transpose
        Y = Mat_transpose# - mur_transpose
        from sklearn.metrics.pairwise import cosine_similarity
        # sim_item_cosine = 1 - cosine_similarity(X, Y)
        sim_item_cosine = cosine_similarity(X, Y)
        # sim_item_cosine = Util.adjusted_cosine(Mat, Mat, mean_user_ratings)

        print "Process ends in {} seconds".format(time.time() - start_time)
        print "Sim matrix properties: {}".format(sim_item_cosine.shape)

        return Mat, sim_item_cosine

    def _crossValidation(self, n_folds=5):
        """
        Measures the RMSE of predictions using the generated similarity matrix (with adjusted cosine similarity)
        and 5-fold cross validation
        Args:
            n_folds: number of k-folds to perform. Default: 5

        """
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
                M[e[0] - 1][e[1] - 1] = e[2]

            true_rate = []
            pred_rate_cosine = []

            for e in test:
                user = e[0]
                item = e[1]
                true_rate.append(e[2])

                # default value: mean rating
                pred_cosine = 3.0

                # item-based
                if np.count_nonzero(M[:, item - 1]):
                    sim_cosine = self.__sim_matrix[item - 1]
                    # get indices of users with non zero rating value
                    ind = (M[user - 1] > 0)
                    normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
                    if normal_cosine > 0:
                        pred_cosine = np.dot(sim_cosine, M[user - 1]) / normal_cosine

                # clip predition under the range [0, 5]
                pred_cosine = np.clip(pred_cosine, 0, 5)

                # print str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_cosine)# + "\t" + str(pred_jaccard) + "\t" + str(pred_pearson)
                pred_rate_cosine.append(pred_cosine)

            rmse_cosine.append(sqrt(mean_squared_error(true_rate, pred_rate_cosine)))

        # print str(sqrt(mean_squared_error(true_rate, pred_rate_cosine))) # + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))

        rmse_cosine = sum(rmse_cosine) / float(len(rmse_cosine))

        print "Cross-validation process finished in {} seconds".format(time.time() - start_time)
        print "Final RMSE value with {}-fold cross validation: {}".format(n_folds,
                                                                          rmse_cosine)  # + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson)

        # f_rmse = open("rmse_item.txt", "w")
        # f_rmse.write(str(rmse_cosine))  # + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

        # rmse = [rmse_cosine]  # , rmse_jaccard, rmse_pearson]
        # req_sim = rmse.index(min(rmse))

        # print req_sim
        # f_rmse.write(str(req_sim))
        # f_rmse.close()

    def get_transition_matrix(self, generate=False, beta=0.9):
        """
        Computes the transition matrix using formulation in:

        @inproceedings{yildirim2008random,
            title={A random walk method for alleviating the sparsity problem in collaborative filtering},
            author={Yildirim, Hilmi and Krishnamoorthy, Mukkai S},
            booktitle={Proceedings of the 2008 ACM conference on Recommender systems},
            pages={131--138},
            year={2008},
            organization={ACM}
        }

        Args:
            generate:
            beta:

        Returns:

        """
        if generate:
            # sum_Sim = np.sum(self.__sim_matrix)
            sum_Sim = np.sum(self.__sim_matrix, axis=1)
            # self.__tr_matrix = csr_matrix(self.__sim_matrix.shape, dtype=np.float64)
            self.__tr_matrix = np.zeros_like(self.__sim_matrix)
            for i in range(self.__sim_matrix.shape[1]):
                self.__tr_matrix[i, ...] = (((beta * self.__sim_matrix[i, ...]) / sum_Sim[i]) if sum_Sim[i] > 0 else 0)\
                                           + ((1. - beta) / self.__sim_matrix.shape[1])
            # self.__tr_matrix = (((beta * self.__sim_matrix) / sum_Sim) if sum_Sim > 0 else 0) \
            #                    + ((1. - beta) / self.__shape[1])
            np.savetxt("trmatrix.txt", self.__tr_matrix)

        return self.__tr_matrix

    def get_transitions_per_item(self, item_id):
        """
        Gets the transition probability for next items given the current item
        Args:
            item_id:

        Returns:

        """

        return self.get_transition_matrix()[item_id - 1]

    def _compute_rankings(self, alpha=0.1, scale=False):
        """
        Compute ratings (or rankings) based on the transition probability matrix and scaling, using formuation in:

        @inproceedings{yildirim2008random,
            title={A random walk method for alleviating the sparsity problem in collaborative filtering},
            author={Yildirim, Hilmi and Krishnamoorthy, Mukkai S},
            booktitle={Proceedings of the 2008 ACM conference on Recommender systems},
            pages={131--138},
            year={2008},
            organization={ACM}
        }


        Args:
            alpha: probability of continuing the random walk. The lower the alpha value is, better predictions when
            random walk is based on the adjusted cosine similarity

        Returns:

        """

        # random_walk_length = np.ones_like(self.__tr_matrix) - alpha * self.__tr_matrix
        random_walk_length = csr_matrix(1 - alpha * self.__tr_matrix).todense()
        # random_walk_length = csc_matrix(1 - alpha * self.__tr_matrix)
        # P_hat = alpha * self.__tr_matrix * np.linalg.pinv(random_walk_length)
        # from scipy.sparse.linalg import inv
        # inv_random_walk_length = inv(random_walk_length)
        inv_random_walk_length = np.linalg.pinv(random_walk_length)
        # np.savetxt("inv.txt", inv_random_walk_length)
        P_hat = alpha * self.__tr_matrix * csr_matrix(inv_random_walk_length)
        # np.savetxt("p_hat.txt", P_hat)
        # self.__ranking_matrix = np.dot(self.__ratings_matrix, P_hat)
        self.__ranking_matrix = self.__ratings_matrix * csr_matrix(P_hat)

        # return np.clip(self._scale_rows(self.__ranking_matrix), 0.0, 5.0) if scale else self.__ranking_matrix.toarray()
        return self._scale_rows(self.__ranking_matrix) if scale else self.__ranking_matrix.toarray()

    def _scale_rows(self, matrix):
        """
        Scales matrix rows linearly considering the max value rated as 5
        Args:
            matrix:

        Returns:

        """
        m = matrix.toarray()
        max_indexes = np.argmax(m, axis=1)
        # scales = [5. / matrix[row][max_indexes[row]] for row in np.arange(len(max_indexes))]
        scales = [5. / m[crow][max_indexes[crow]] for crow in np.arange(len(max_indexes))]
        # scales = [5. - m[crow][max_indexes[crow]] for crow in np.arange(len(max_indexes))]
        # scales = np.expand_dims(scales, axis=0)
        scales = np.expand_dims(scales, axis=1)
           
        return np.round(scales * matrix.toarray())

    def get_rankings_per_user(self, user_id):
        """
        Gets the predicted ratings per user
        Args:
            user_id:

        Returns:

        """
        return self.__ranking_matrix[user_id - 1]

    def predict_item_rating(self, user_id, item_id):
        """
        predicts item rating per user by using the rating and similarity matrices
        :param item_id:
        :param user_id:
        :return:
        """
        pred_cosine = 0.
        if np.count_nonzero(self.__ratings_matrix[:, item_id - 1]):
            sim_cosine = self.__sim_matrix[item_id - 1]
            # get indices of users with non zero rating value
            ind = (self.__ratings_matrix[user_id - 1] > 0)
            normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
            if normal_cosine > 0:
                pred_cosine = np.dot(sim_cosine, self.__ratings_matrix[user_id - 1]) / normal_cosine

        return np.clip(pred_cosine, 0, 5)

    def predictRating(self, toBeRated):
        """
        Makes prediction based on:
            1) the user rating matrix and the similarity matrix
            2) random walk ranking matrix
        Args:
            toBeRated:

        Returns:

        """

        # f = open("toBeRated.csv","r")

        pred_rate = []
        pred_rate_by_rank = []
        y_true = []

        # fw = open('result2.csv','w')
        fw_w = open('data/predictions.csv', 'w')
        fw_r = open('data/predictions-by-Rank.csv', 'w')

        eval_ = []
        eval_by_rank = []

        l = len(toBeRated["user"])
        # last_user = None
        for e in range(l):
            user = toBeRated["user"][e]
            item = toBeRated["item"][e]
            true_label = toBeRated["true_label"][e]
            y_true.append(true_label)

            pred = 3.0

            # item-based
            if np.count_nonzero(self.__ratings_matrix[:, item - 1]):
                sim = self.__sim_matrix[item - 1]
                ind = (self.__ratings_matrix[user - 1] > 0)
                # print "ind -> {}".format(ind)
                # ind[item-1] = False
                normal = np.sum(np.absolute(sim[ind]))
                if normal > 0:
                    pred = np.round(np.dot(sim, self.__ratings_matrix[user - 1]) / normal)

            if pred < 0:
                pred = 0

            if pred > 5:
                pred = 5

            pred_rate.append(int(pred))


            # predictions by ranking matrix
            pred_by_rank = self.get_rankings_per_user(user - 1)[item -1]
            pred_rate_by_rank.append(int(pred_by_rank))

            # print str(user) + "," + str(item) + "," + str(pred)
            # fw.write(str(user) + "," + str(item) + "," + str(pred) + "\n")
            fw_w.write(str(true_label) + " " + str(pred) + "\n")
            fw_r.write(str(true_label) + " " + str(pred_by_rank) + "\n")

            # if e <= l and (toBeRated["user"][e + 1] if e+1 < l else None) != user:
            #     if e == 7332:
            #         precision_score(y_true, pred_rate, average='micro')
            #     precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_rate, beta=0.5, average='micro')
            #     eval_.append([precision, recall, f1])
            #     precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_rate_by_rank, beta=0.5,
            #                                                                average='micro')
            #     eval_by_rank.append([precision, recall, f1])
            #     pred_rate = []
            #     pred_rate_by_rank = []
            #     y_true = []

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_rate, beta=0.5, average='micro')
        # precision = eval_[0] / len(eval_)
        # recall = eval_[1] / len(eval_)
        # f1 = eval_[2] / len(eval_)
        print "Predictions Summary - Evaluation averaged per user"
        print "==================="
        print "Precision: {} -- Recall: {} -- F1Score: {}\n\n".format(precision, recall, f1)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_rate_by_rank, beta=0.5, average='micro')
        # precision = eval_[0] / len(eval_by_rank)
        # recall = eval_[1] / len(eval_by_rank)
        # f1 = eval_[2] / len(eval_by_rank)
        print "Predictions by Rank Summary - Evaluation averaged per user"
        print "==================="
        print "Precision: {} -- Recall: {} -- F1Score: {}".format(precision, recall, f1)

        # fw.close()
        fw_w.close()
        fw_r.close()


def readingFile(filename, split=','):
    f = open(filename, "r")
    data = []
    for row in f:
        r = row.split(split)
        e = [int(r[0]), int(r[1]), int(r[2])]
        data.append(e)
    return data


if __name__ == "__main__":
    field_split = sys.argv[2]
    field_split = "\t" if field_split[0:1] == '\\' else field_split
    recommend_data = readingFile(sys.argv[1], split=field_split)
    toBeRated = {"user": [], "item": [], "true_label": []}
    if field_split == '\t':

        inst = TransitionProbability(train=True, raw_data=recommend_data, shape=(943, 1682))
        if len(sys.argv) > 3:
            f = open(sys.argv[3], "r")
            for row in f:
                r = row.split('\t')
                toBeRated["item"].append(int(r[1]))
                toBeRated["user"].append(int(r[0]))
                toBeRated["true_label"].append(int(r[2]))

            f.close()
            inst.predictRating(toBeRated)
            # for alpha in np.arange(0.325, 0, -0.025):
            #     for beta in np.arange(0.6, 1, 0.1):
            #         # inst = TransitionProbability(train=False, raw_data=recommend_data, shape=(943, 1682))
            #
            #         print "=========================================="
            #         print "=========================================="
            #         print "Alpha: {} - Beta: {}".format(alpha, beta)
            #         print "=========================================="
            #         print "=========================================="
            #
            #         inst = TransitionProbability(train=True, raw_data=recommend_data, shape=(943, 1682), alpha=alpha, beta=beta)
            #         # crossValidation(recommend_data)
            #         if len(sys.argv) > 2:
            #             f = open(sys.argv[2], "r")
            #             toBeRated = {"user": [], "item": [], "true_label": []}
            #             for row in f:
            #                 r = row.split('\t')
            #                 toBeRated["item"].append(int(r[1]))
            #                 toBeRated["user"].append(int(r[0]))
            #                 toBeRated["true_label"].append(int(r[2]))
            #
            #             f.close()
            #             inst.predictRating(toBeRated)

    else:
        from sklearn.cross_validation import train_test_split
        _, X_test = train_test_split(recommend_data, test_size=0.2, random_state=0)
        inst = TransitionProbability(train=True, raw_data=recommend_data, shape=(6040, 3883))
        # for r in X_test:
        #     toBeRated["item"].append(int(r[1]))
        #     toBeRated["user"].append(int(r[0]))
        #     toBeRated["true_label"].append(int(r[2]))
        # start_time = time.time()
        # inst.predictRating(toBeRated)
        # print "prediction processing time: {} seconds".format(time.time() - start_time)


