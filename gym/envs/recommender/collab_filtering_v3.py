"""
Collaborative Filtering environment for RL
"""

import logging
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.recommender.util import *
from six import StringIO
import sys


logger = logging.getLogger(__name__)


class CollaborativeFilteringV3(gym.Env):

    metadata = {
        'render.modes': ['human', 'ansi'], #'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, properties):

        self.__init_done = False
        # data manager for simulation
        self.__data = None
        self.properties = register(properties) #temporal. JUST FOR TESTING PURPOSES
        self._configure()
        self._seed()

        sample_item = self.__data.find_one(type="items")
        embeddings = np.array(sample_item['embeddings']).ravel()
        self.__emb_size = embeddings.size
        other_features = np.array(sample_item['other_feat']).ravel()
        self.__other_size = other_features.size

        # sample_user = self.__data.find_one(type="ratings")
        self.user_bounds = self.__data.get_bounds(type="ratings")

        # exploration strategy for episodes
        self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.np_random.shuffle(self.user_exploration)

        # self.__current_user = sample_user["user_id"]
        self.__previous_user = None
        self.__current_user = None
        self.obs_id = sample_item["_id"]
        self.__embeddings = np.ones(shape=self.__emb_size)
        self.__other_feat = np.ones(shape=self.__other_size)

        self.__true_positives = self.__true_negatives = self.__false_positives = self.__false_negatives = 0.
        self.__selected_items = []

        low = np.concatenate([-1 * np.ones(self.__emb_size), np.zeros(self.__other_size)]).ravel()
        high = np.concatenate([np.ones(self.__emb_size), np.ones(self.__other_size)]).ravel()
        # observation space defined as the previous recommended item
        self.observation_space = spaces.Box(low, high)

        # action space defined as the set of available items to recommend
        low, high = self.__data.get_bounds(type="items") # if we want to represent the action space to the item_id in the set
        vector_length = len("{0:b}".format(high))
        # low = "{0:0{1}b}".format(low, vector_length)
        # high = "{0:b}".format(high)
        # low = np.array([float(low[i]) for i in range(vector_length)])
        # high = np.array([float(high[i]) for i in range(vector_length)])
        low = np.zeros(vector_length)
        high = np.ones(vector_length)
        # self.action_space = spaces.Box(np.array([1. * low]), np.array([1. * high])) # enclosed by squared brackets only when the action space is set to item id set
        self.action_space = spaces.Box(low, high)

        # if self.properties.kwargs['use_mongodb']:
        #     self._get_tr_stats(low, high)
        self.viewer = None

        self.__init_done = True

    # def _get_tr_stats(self, low, high):
    #     import time
    #     start_time =time.time()
    #     p = np.zeros(shape=high ** 2)
    #     c = 0
    #     for i in np.arange(start=low, stop=high):
    #         for o in np.arange(start=low, stop=high):
    #             p[c] = self.__data.compute_transition(i, o)
    #             c += 1
    #
    #     end_time = time.time() - start_time
    #     print("mean {} std {}".format(np.mean(p), np.std(p)))
    #     print("Time processing tr matrix statistics: {} seconds".format(end_time))

    def _configure(self, display=None):
        if not self.__init_done:
            self.__data = MongoDB(kwargs=self.properties.kwargs) if self.properties.kwargs['use_mongodb'] \
                else FileSystemData(kwargs=self.properties.kwargs)
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_data(self):
        return self.__data

    def _reset(self):
        # set user to work under an episode
        if len(self.user_exploration) == 0:
            self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.__current_user = self.user_exploration[0]
        self.user_exploration = self.user_exploration[1:]

        self.__selected_items = []

        # random item state
        self.obs_id, item = self._get_guided_random_item()
        self.obs_id = int(self.obs_id)
        self.__embeddings = np.array(item['embeddings']).ravel()
        self.__other_feat = np.array(item['other_feat']).ravel()
        self.__true_positives = self.__true_negatives = self.__false_positives = self.__false_negatives = 0.
        return self._get_obs()

    def _get_obs(self):
        '''
        Gives the actual representation of an observation as a joint vector of word embeddings and other features
        Returns:

        '''
        return np.concatenate([self.__embeddings, self.__other_feat]).ravel()

    def _get_guided_random_item(self):
        '''
        Gets a random item from a likely subset of items
        Returns:

        '''
        # get subset of good actions for the user to give
        subset_size = self.properties.kwargs['expl_subset_limit']
        # take the subset_size number of likely items to be selected by the current user
        # sum 1 unit as it returns an array of indexes positions
        user_items = 1 + self.__data.get_user_items(self.__current_user, limit=subset_size)
        random_item = int(np.floor(self.np_random.uniform(low=0, high=subset_size)))
        # random item state
        random_item_id = user_items[random_item]['item_id'] if self.properties.kwargs['use_mongodb'] else user_items[random_item]
        # print "random item id in guided exploration: {}".format(random_item_id)
        return random_item_id, self.__data.find_one(query={"_id": random_item_id}, type="items")

    def _step(self, action):
        """

        :param action:
        :return:
        """

        done = False
        p_end_episode = 0.
        reward = 0.0

        # FOR DISCRETE ACTION SPACE
        # # Hack to obtain the action as an integer that correspond to an existing item id in the item set
        action = np.round(action)



        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # FOR DISCRETE ACTION SPACE
        action_id = int("".join([str(int(a)) for a in action]), 2)

        low, high = self.__data.get_bounds(type="items")

        if low <= action_id <= high:
            is_chosen, p_action, pred_rating, action_id = self.__data.compute_transition(self.__current_user,
                                                                                         self.obs_id,
                                                                                         action_id)
        else:
            is_chosen = False

        info = {}

        item_info = {}
        if is_chosen: # (simulated) user selected the item (with max probability)
            self.__true_positives += 1
            info["item selected with P(X)"] = p_action
            p_end_episode = 0.1 # the episode ends with probability 0.1
            action_id = int(action_id)
            item_info = {"previous": self.obs_id, "item": action_id, "reward": pred_rating,
                         "info": {"rating": pred_rating, "recommended": "X", "random": "-"}}
        else: # user selects another item randomly
            self.__false_positives += 1
            p_end_episode = 0.2 # the episode ends with probability 0.2
            info["Guided exploration"] = self.properties.kwargs["guided_exploration"]
            if self.properties.kwargs["guided_exploration"]: # explore from a subset of likely items for the user
                _, random_item = self._get_guided_random_item()
            else: # explore uniformly from the whole set of items
                random_item = self.__data.find_one(type="items")

            action_id = random_item["_id"]
            info["item randomly selected"] = action_id
            # FOR DISCRETE ACTION SPACE
            _, p_action, pred_rating, _ = self.__data.compute_transition(self.__current_user, self.obs_id, action_id)

            info["item selected with P(X)"] = p_action
            item_info = {"previous": self.obs_id, "item": action_id, "reward": 0.,
                         "info": {"rating": pred_rating, "recommended": "-", "random": "X"}}

            # set reward to ZERO as the user selects an item randomly
            pred_rating = 0.

        reward = pred_rating

        self.__selected_items.append(item_info)

        # TODO: calculate false negatives and true negatives
        info["Precision for current user"] = self.__true_positives / (self.__true_positives + self.__false_positives)

        if self.np_random.uniform() < p_end_episode:
            done = True
            self.__previous_user = self.__current_user
            # TODO: calculate MAP for cumulative Precision over users (episodes)

        self.obs_id = action_id
        # obs = self.__data.find_one(query={"_id": self.obs_id}, type="items")
        # self.__embeddings = np.array(obs['embeddings']).ravel()
        # self.__other_feat = np.array(obs['other_feat']).ravel()
        _, self.__embeddings, self.__other_feat = self.__data.get_item_representation(self.obs_id)

        return self._get_obs(), round(reward), done, info

    def _render(self, mode='human', close=False):
        if close:
            #Nothing to do
            return

        output = StringIO() if mode == 'ansi' else sys.stdout

        # head  = "Running episode with user {0:06d}\n".format(self.__current_user)
        head  = "================================\n"
        head += "====Item recommended so far=====\n"
        head += "================================\n"
        head += "User: {}\n".format(self.__previous_user)
        head += "================================\n"
        head += "\n"
        output.write(head)
        body = "  #  || Prev  Item ||  Item  || Rating || Reward || Recom || Random \n"
        i = 0
        reward = 0.
        for item in self.__selected_items:
            info = item["info"]
            reward += item["reward"]
            body += "{0:05d}||   {1:06d}   || {2:06d} ||  {3:.2f}  ||  {4:.2f}  ||___{5}___||___{6}___ \n"\
                .format(i, item["previous"], item["item"], info["rating"], item["reward"], info["recommended"],
                        info["random"])
            i += 1


        output.write(body if i > 0 else "Initializing....\n")

        precision = self.__true_positives / (self.__true_positives + self.__false_positives) \
            if (self.__true_positives + self.__false_positives) > 0 else 0.

        stats  = "Precision at {0:05d} : {1:.3f}\n".format(i, precision)
        stats += "Cummulative Reward: {0:.2f}\n".format(reward)

        output.write(stats)

        return output
