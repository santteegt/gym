from gym.spaces import box
from gym.spaces import prng
import numpy as np


class SimBox(box.Box):

    def __init__(self, low, high, shape=None, env=None, top_n=20):
        assert env is not None, "Environment cannot be None"
        assert callable(getattr(env, "get_data", None)), "If no custom data is available for sampling, it is " \
                                                         "better to use Box class directly"
        self.__data = env.get_data()
        self.__env = env
        self.top_n = top_n
        super(SimBox, self).__init__(low, high, shape)

    def sample(self):
        """
        Sample action space by selecting from a likely subset of items
        :param observation: current observation
        :param top_n:
        :return:
        """
        top_n_range = range(self.top_n)
        prng.np_random.shuffle(top_n_range)
        sample = top_n_range[0]
        sample_item_id = self.__data.get_likely_items_per_state(self.__env.obs_id)[sample]
        sample_item, _, _ = self.__data.get_item_representation(sample_item_id)

        return sample_item

