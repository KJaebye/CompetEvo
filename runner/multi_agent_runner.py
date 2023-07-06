from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.evo_learner import EvoLearner

class MultiAgentRunner(BaseRunner):
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True) -> None:
        super().__init__(cfg, logger, dtype, device, num_threads=num_threads, training=True)

    def setup_learner(self):
        """ Learners are corresponding to agents. """
        self.learners = {}
        for i, agent in self.env.agents.items():
            if "evo" in agent.team:
                self.learners[i] = EvoLearner(self.cfg, self.dtype, self.device, agent)
            else:
                self.learners[i] = Learner(self.cfg, self.dtype, self.device, agent)

    def optimize(self):
        

