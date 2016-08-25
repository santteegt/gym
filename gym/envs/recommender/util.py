from pymongo import MongoClient, ASCENDING, DESCENDING
import os
import pandas
from transition_matrix import TransitionProbability
from gym import error
import numpy as np

class Data(object):

    data_dir = 'data'

    def __init__(self, random):
        self.random = random

    def get_next_action(self):
        pass

    def get_transition_p(self, term1, term2):
        pass

    def find_one(self, query=None, type="items"):
        pass

    def get_bounds(self, type="items"):
        pass

    def get_user_items(self, user_id, limit=5):
        pass

    def compute_transition(self, prev_obs, obs, embeddings=None):
        pass

class MongoDB(Data):
    def __init__(self, random=None, kwargs=None):
        super(MongoDB, self).__init__(random)
        try:
            self.__client = MongoClient()
            # Database
            self.__db = self.__client[kwargs['mongo_db']]
            self.__items = self.__db[kwargs['items_collection']]
            # TODO: it needs to be updated to be generated and read from new class. See FileSystemData class
            # self.__trmatrix = self.__db[kwargs['trmatrix_collection']]
            self.__user_ratings = self.__db[kwargs['ratings_collection']]
        except Exception as e:
            raise error.Error("Error while trying to load data from MongoDB: {}".format(e))

    def find_one(self, query=None, type="items"):
        assert type == "items" or type == "ratings", \
            "Wrong type value for query. Given: {}. Possible Values = [items, ratings]".format(type)

        collection = self.__items if type == "items" else self.__user_ratings
        return collection.find_one(query if query is not None else {})

    def get_bounds(self, type="items"):
        assert type == "items" or type == "ratings", \
            "Wrong type value for query. Given: {}. Possible Values = [items, ratings]".format(type)

        collection = self.__items if type == "items" else self.__user_ratings
        field_id = "_id" if type == "items" else "user_id"
        max = collection.find().sort(field_id, DESCENDING).limit(1).next()[field_id]
        min = collection.find().sort(field_id, ASCENDING).limit(1).next()[field_id]
        return min, max

    def get_user_items(self, user_id, limit=5):
        return self.__user_ratings.find().sort({"rating": 1}).limit(limit)

    # def compute_transition(self, prev_obs_id, obs_id):
    #     pair_id = str(prev_obs_id) + "|" + str(obs_id)
    #     pair_rs = self.__trmatrix.find({"_id": pair_id})
    #     norm_rs = self.__trmatrix.find({"_id": str(prev_obs_id)})
    #     # return normalized bigram counts
    #     return pair_rs.next()["count"] / norm_rs.next()["count"] if norm_rs.count() > 0 and pair_rs.count() > 0 else 0.0

class FileSystemData(Data):
    def __init__(self, kwargs, random=None):
        local_properties = kwargs['local']
        path = local_properties['path']
        users, items, ratings = self._read_info(os.path.join(path, local_properties['info']))
        self.__stats = {"users": users, "items": items,  "ratings": ratings}
        self.__items = self._read_csv_data(os.path.join(path, local_properties['items_collection']))
        self.__user_ratings = self._read_numpy_array(os.path.join(self.data_dir, local_properties['ratings_collection']))
        self.__trmatrix = TransitionProbability(train=local_properties['train'], raw_data=self.__user_ratings,
                                                shape=(users, items), data_dir=self.data_dir)

    def _read_csv_data(self, file_path, separator=',', header=None):
        return pandas.read_csv(file_path, sep=separator)

    def _read_numpy_array(self, filename):
        return np.load(filename, "r", allow_pickle=False)

    def _read_info(self, file_path):
        f = open(file_path, "r")
        users = items = ratings = None
        i = 0
        for row in f:
            if i == 0: users = int(row.split(" ")[0])
            elif i == 1: items = int(row.split(" ")[0])
            elif i == 2:
                ratings = int(row.split(" ")[0])
                break
            i += 1
        return users, items, ratings

    def find_one(self, query=None, type="items"):
        assert type == "items" or type == "ratings", \
            "Wrong type value for query. Given: {}. Possible Values = [items, ratings]".format(type)
        if type == "items":
            rs = self.__items.sample() if query is None else \
                self.__items[self.__items[query.keys()[0]] == query.values()[0]]
            # TODO: remove this hack and read arrays in panda accordingly
            item = {}
            item['_id'] = rs['_id'].values[0]
            item['embeddings'] = np.fromstring(rs['embeddings'].values[0][1:][:-1], sep=" ")
            item['other_feat'] = np.fromstring(rs['other_feat'].values[0][1:][:-1], sep=" ")
            return item
        else:
            rating = None
            if query is None:
                rows = self.__user_ratings.shape[0] - 1
                rating = self.__user_ratings[np.round(np.random.uniform(0, rows))]
            else: # TODO
                rating = rating
            return rating

    def get_bounds(self, type="items"):
        assert type == "items" or type == "ratings", \
            "Wrong type value for query. Given: {}. Possible Values = [items, ratings]".format(type)
        if type == "items":
            ids = self.__items["_id"]
            return ids.min(), ids.max()
        else:
            return 1, self.__user_ratings.shape[0]

    def get_user_items(self, user_id, limit=5):
        return self.__user_ratings[user_id - 1].argsort()[::-1][:limit]

    def compute_transition(self, user_id, prev_obs, obs, embeddings=None):
        '''
        Computes the transition probability of an item given the current state and returns its value and simulated user
        rating if it is likely chosen
        Args:
            user_id:
            prev_obs:
            obs:

        Returns:

        '''
        if embeddings is not None:
            embed_feat = obs[:embeddings[0]]
            other_feat = obs[embeddings[0]:].astype(np.int)
            item = self.__items[(self.__items["embeddings"] == str(embed_feat))
                                & (self.__items["other_feat"] == str(other_feat))]
            assert item.values.size > 0, "The query is not returning features for an existing item"
            obs_id = item["_id"].values[0]
        else:
            obs_id = obs


        # get the item transition with max probability
        item_transitions = self.__trmatrix.get_transitions_per_item(prev_obs)
        max_p_index = np.argmax(item_transitions)
        rating = self.__user_ratings[user_id - 1][obs_id - 1]
        # get existing rating, otherwise, select the rating from predictions
        rating = rating if rating > 0 else self.__trmatrix.get_rankings_per_user(user_id)[obs_id - 1]

        return max_p_index + 1 == obs_id, item_transitions[obs_id - 1], np.round(rating), obs_id


# just for testing purposes
class register():
    def __init__(self, properties):
        self.id='CollaborativeFiltering-v0'
        self.entry_point='gym.envs.recommender:Collaborative_Filtering'
        self.reward_threshold=195.0
        self.timestep_limit=200
        self.trials=100 #default value
        # local_only=False,
        # self.kwargs={'use_mongodb': False, 'mongo_db': 'recommender', 'items_collection': 'items_movielens100k',
        #         'trmatrix_collection': 'trmatrix_movielens100k', "ratings_collection": 'ratings_movielens100k',
        #         'expl_subset_limit': 5, "guided_exploration": False,
        #         'local': {'train': False, 'path': '/Users/santteegt/Downloads/ml-100k',
        #                   'info': 'u.info', 'items_collection': 'items_collection.csv',
        #                   'trmatrix_collection': 'trmatrix.npy', 'ratings_collection': 'rmatrix.npy',
        #                   'predicted_ratings': 'rankmatrix.npy'}
        #         }
        self.kwargs = properties
        self.nondeterministic=False