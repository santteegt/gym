"""
Collaborative Filtering environment for RL
"""

import logging
import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.recommender.util import *


logger = logging.getLogger(__name__)

class CollaborativeFiltering(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
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
        emb_size = embeddings.size
        other_features = np.array(sample_item['other_feat']).ravel()
        other_size = other_features.size

        # sample_user = self.__data.find_one(type="ratings")
        self.user_bounds = self.__data.get_bounds(type="ratings")

        self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.np_random.shuffle(self.user_exploration)

        # self.__current_user = sample_user["user_id"]
        self.__current_user = None
        self.__obs_id = sample_item["_id"]
        self.__embeddings = np.ones(shape=emb_size)
        self.__other_feat = np.ones(shape=other_size)

        self.__true_positives = 0
        self.__true_negatives = 0
        self.__false_positives = 0
        self.__false_negatives = 0

        low = np.concatenate([-1 * np.ones(emb_size), np.zeros(other_size)]).ravel()
        high = np.concatenate([np.ones(emb_size), np.ones(other_size)]).ravel()
        # observation space defined as the previous recommended item
        self.observation_space = spaces.Box(low, high)
        # action space defined as the set of available items to recommend
        low, high = self.__data.get_bounds(type="items") # if we want to represent the action space to the item_id in the set
        self.action_space = spaces.Box(np.array([low]), np.array([high])) # enclosed by squared brackets only when the action space is set to item id set
        # self.action_space = spaces.Box(low, high)

        # if self.properties.kwargs['use_mongodb']:
        #     self._get_tr_stats(low, high)
        self.viewer = None

        # # Just need to initialize the relevant attributes
        # self._configure()

        # =====

        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5 # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        #
        # # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        #
        # # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        # high = np.array([
        #     self.x_threshold * 2,
        #     np.finfo(np.float32).max,
        #     self.theta_threshold_radians * 2,
        #     np.finfo(np.float32).max])
        #
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high)
        #
        # self._seed()
        # self.reset()
        # self.viewer = None
        #
        # self.steps_beyond_done = None
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

    def _reset(self):
        # set user to work under an episode
        if len(self.user_exploration) == 0:
            self.user_exploration = np.arange(self.user_bounds[0], self.user_bounds[1] + 1)
        self.__current_user = self.user_exploration[0]
        self.user_exploration = self.user_exploration[1:]


        # random item state
        self.__obs_id, item = self._get_guided_random_item()
        self.__embeddings = np.array(item['embeddings']).ravel()
        self.__other_feat = np.array(item['other_feat']).ravel()
        self.__true_positives = self.__true_negatives = self.__false_positives = self.__false_negatives = 0
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
        print "random item id in guided exploration: {}".format(random_item_id)
        return random_item_id, self.__data.find_one(query={"_id": random_item_id}, type="items")

    def _step(self, action):

        done = False
        p_end_episode = 0.
        reward = 0.0
        # Hack to obtain the action as an integer that correspond to an existing item id in the item set
        action = np.round(action)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        is_chosen, p_action, pred_rating = self.__data.compute_transition(self.__current_user, self.__obs_id, int(action))
        info = {}

        if is_chosen: # (simulated) user selected the item
            self.__true_positives += 1
            info["item selected with P(X)"] = p_action
            p_end_episode = 0.1 # the episode ends with probability 0.1
            action = int(action)
        else: # user selects another item randomly
            self.__false_positives += 1
            p_end_episode = 0.2 # the episode ends with probability 0.2
            info["Guided exploration"] = self.properties.kwargs["guided_exploration"]
            if self.properties.kwargs["guided_exploration"]: # explore from a subset of likely items for the user
                _, random_item = self._get_guided_random_item()
            else: # explore uniformly from the whole set of items
                random_item = self.__data.find_one(type="items")

            action = random_item["_id"]
            info["item randomly selected"] = action
            _, p_action, pred_rating = self.__data.compute_transition(self.__current_user, self.__obs_id, action)
            info["item selected with P(X)"] = p_action

        reward = pred_rating

        # TODO: calculate false negatives and true negatives
        info["Precision for current user"] = self.__true_positives / (self.__true_positives + self.__false_positives)

        if self.np_random.uniform() < p_end_episode:
            done = True
            # TODO: calculate MAP for cumulative Precision over users (episodes)

        self.__obs_id = action
        obs = self.__data.find_one(query={"_id": self.__obs_id}, type="items")
        self.__embeddings = np.array(obs['embeddings']).ravel()
        self.__other_feat = np.array(obs['other_feat']).ravel()

        return self._get_obs(), round(reward), done, info






        # state = self.state
        # x, x_dot, theta, theta_dot = state
        # force = self.force_mag if action==1 else -self.force_mag
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        # xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # x  = x + self.tau * x_dot
        # x_dot = x_dot + self.tau * xacc
        # theta = theta + self.tau * theta_dot
        # theta_dot = theta_dot + self.tau * thetaacc
        # self.state = (x,x_dot,theta,theta_dot)
        # done =  x < -self.x_threshold \
        #         or x > self.x_threshold \
        #         or theta < -self.theta_threshold_radians \
        #         or theta > self.theta_threshold_radians
        # done = bool(done)
        #
        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        #     self.steps_beyond_done += 1
        #     reward = 0.0
        #
        # return np.array(self.state), reward, done, {}

    def _render(self, mode='human', close=False):
        return []
        # if close:
        #     if self.viewer is not None:
        #         self.viewer.close()
        #         self.viewer = None
        #     return
        #
        # screen_width = 600
        # screen_height = 400
        #
        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        # carty = 100 # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * 1.0
        # cartwidth = 50.0
        # cartheight = 30.0
        #
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
        #     l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        #     axleoffset =cartheight/4.0
        #     cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        #     pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        #     pole.set_color(.8,.6,.4)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = rendering.make_circle(polewidth/2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(.5,.5,.8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = rendering.Line((0,carty), (screen_width,carty))
        #     self.track.set_color(0,0,0)
        #     self.viewer.add_geom(self.track)
        #
        # x = self.state
        # cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        #
        # return self.viewer.render(return_rgb_array = mode=='rgb_array')