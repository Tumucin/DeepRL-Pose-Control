from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np
from pyquaternion import Quaternion

class CUSTOMCALLBACK(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0, config=None):
        super(CUSTOMCALLBACK, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.currentWorkspace = 0 # 0 means that it is fixed 
        self.currentIteration = 0
        self.workspaceLen = None
        self.config = config
        self.testingEnv = gym.make(self.config['envName'], render=False)
        self.testingEnv.env.robot.config = self.config
        self.testingEnv.env.task.config = self.config
        self.testingEnv.env._max_episode_steps = self.config['max_episode_steps']
        self.rmse = 0
        self.mae = 0
        self.avgJntVel = 0
        self.counterTesting = 1
        
        

    def evaluatePolicy(self, numberOfSteps, model, env):
        mae = 0.0
        squaredError = 0.0
        successRate1 = 0.0
        successRate5 = 0.0
        avgJntVel = 0.0
        avgQuaternionDistance = 0.0
        avgQuaternionAngle = 0.0

        for step in range(numberOfSteps):
            obs = env.reset()
            print("step in CustomCallBack.py: ", step)
            done = False
            
            episode_reward = 0.0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            error = abs(obs['achieved_goal'] - obs['desired_goal'])
            mae = np.linalg.norm(error) + mae
            squaredError += np.sum(error**2)
            avgJntVel = np.linalg.norm(env.robot.finalAction) + avgJntVel
            d1 = env.robot.goalFrame.M.GetQuaternion()
            c1 = env.sim.get_link_orientation(env.sim.body_name, self.config['ee_link'])
            desiredQuaternion = Quaternion(d1[3], d1[0], d1[1], d1[2])
            currentQuaternion = Quaternion(c1[3], c1[0], c1[1], c1[2])
            avgQuaternionDistance+=Quaternion.distance(desiredQuaternion, currentQuaternion)
            quaternionError = desiredQuaternion*currentQuaternion.conjugate
            avgQuaternionAngle+=quaternionError.angle
            if np.linalg.norm(error) <=0.01:
                successRate1+=1
            
            if np.linalg.norm(error) <=0.05:
                successRate5+=1

                #episode_reward+=reward
        rmse = np.sqrt((squaredError)/(numberOfSteps))
        mae = mae/numberOfSteps
        successRate1 = successRate1/numberOfSteps
        successRate5 = successRate5/numberOfSteps
        avgJntVel = avgJntVel/numberOfSteps
        avgQuaternionDistance=avgQuaternionDistance/numberOfSteps
        avgQuaternionAngle = avgQuaternionAngle/numberOfSteps

        return rmse, mae, successRate1, successRate5, avgJntVel, avgQuaternionDistance, avgQuaternionAngle
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples. After policy update
        """
        flag = False
        self.currentIteration+=1
        
        if self.model.num_timesteps> self.config['evalFreqOnTraining']*self.counterTesting:
            self.counterTesting+=1
            flag = True

        if self.workspaceLen == None:
            self.workspaceLen = len(self.training_env.envs[0].robot.workspacesdict)/2 - 1
        
        if flag and self.currentWorkspace < self.workspaceLen and self.config['CurriLearning']:
            print("current iteration in customCallBack.py:", self.currentIteration)
            print("current time step", self.num_timesteps )

            self.rmse, self.mae, successRate1, successRate5, self.avgJntVel, avgQuaternionDistance, avgQuaternionAngle = self.evaluatePolicy(self.config['testSampleOnTraining'], 
                                                                                                                        self.model, 
                                                                                                                        self.testingEnv)
            print("RMSE in CustomCallBack.py:", self.rmse)
            print("MAE CustomCallBack.py:", self.mae)
            print("Success Rate 1 cm CustomCallBack.py:", successRate1)
            print("Success Rate 5 cm CustomCallBack.py:", successRate5)
            print("Average joint velocities CustomCallBack.py:", self.avgJntVel)
            print("Average quaternion distance:", avgQuaternionDistance)
            print("Average quaternion angle:", avgQuaternionAngle)
            if self.mae < self.config['maeThreshold'] and self.avgJntVel < self.config['avgJntVelThreshold'] :
                # Change workspace 
                self.currentWorkspace+=1
                
                for i in range(self.training_env.num_envs):
                    self.training_env.envs[i].robot.jointLimitLow = self.training_env.envs[i].robot.workspacesdict['W'+str(self.currentWorkspace)+"Low"]
                    self.training_env.envs[i].robot.jointLimitHigh = self.training_env.envs[i].robot.workspacesdict['W'+str(self.currentWorkspace)+"High"]
                    self.training_env.envs[i].task.jointLimitLow = self.training_env.envs[i].robot.jointLimitLow
                    self.training_env.envs[i].task.jointLimitHigh = self.training_env.envs[i].robot.jointLimitHigh
                    self.testingEnv.robot.jointLimitLow = self.training_env.envs[i].robot.workspacesdict['W'+str(self.currentWorkspace)+"Low"]
                    self.testingEnv.robot.jointLimitHigh = self.training_env.envs[i].robot.workspacesdict['W'+str(self.currentWorkspace)+"High"]
                #print("currentWorkspace:", self.currentWorkspace)
                print("training new jointLimitLow:", self.training_env.envs[0].robot.jointLimitLow)
                print("training new jointLimitHigh:", self.training_env.envs[0].robot.jointLimitHigh)
                print("testing new jointLimitLow:", self.testingEnv.robot.jointLimitLow)
                print("testing new jointLimitHigh:", self.testingEnv.robot.jointLimitHigh)

        self.logger.record("rollout/currentWorkspace", self.currentWorkspace)
        self.logger.record("rollout/rmse", self.rmse)
        self.logger.record("rollout/mae", self.mae)
        self.logger.record("rollout/avgJntVel", self.avgJntVel)
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
