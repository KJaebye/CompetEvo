# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class MujocoEnv
#   @author: by Kangyao Huang
#   @created date: 05.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from dm_control.rl.control import Environment
from dm_control.mujoco import Physics
from dm_control.suite.base import Task

class MujocoEnv(Environment):
    """
        Superclass for all MujoCo environments in this proj.
        This class will pass config into environment.
    """

    def __init__(self, physics, task, *args, **kwargs):
        """
        :param physics: Instance of Physics.
        :param task: Instance of Task.
        :param args: ...
        :param kwargs: ...
        """
        super(MujocoEnv, self).__init__(physics, task, *args, **kwargs)


class MujocoPhysics(Physics):
    pass


class MujocoTask(Task):
    def __init__(self, random=None):
        super().__init__(random)
