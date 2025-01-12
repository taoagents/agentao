from typing import List, Dict
import trueskill
import numpy as np
from logging import Logger

from agentao.validator.graders.abstract_grader import GraderInterface, MinerSubmission
from agentao.validator.graders.float_grader import FloatGrader

class TrueSkillGrader(GraderInterface):
    """
    A grader that uses the TrueSkill rating system to grade miners. The 
    ratings are updated based on the performance of the miners in the
    forward loop, and then normalized with a logistic function.
    """
    def __init__(self, logger: Logger):
        self.logger = logger
        self.env = trueskill.TrueSkill()
        self.ratings: Dict[str, trueskill.Rating] = {}
        self.float_grader = FloatGrader(logger)
        self.num_runs = 0
        self.apha = np.log(4) / self.env.beta

    def grade(self, submissions: List[MinerSubmission]) -> List[float]:
        # Initialize any new miners
        for submission in submissions:
            if submission.miner_hotkey not in self.ratings:
                self.ratings[submission.miner_hotkey] = self.env.create_rating()

        # We run the rating system thrice for steadier results when we first
        # initialize the ratings
        if len(submissions) > 1:
            num_runs = 1 if self.num_runs > 5 else 3
            float_scores = self.float_grader.grade(submissions)
            for _ in range(num_runs):
                self.update_ratings(submissions, float_scores)

            self.num_runs += 1

        ratings = []
        mean_score = np.mean([r.mu - 3*r.sigma for r in self.ratings.values()])
        for submission in submissions:
            miner_rating = self.ratings[submission.miner_hotkey]
            miner_rating = miner_rating.mu - 3 * miner_rating.sigma
            ratings.append(1 / (1 + np.exp(-self.apha * (miner_rating - mean_score))))

        return ratings

    def update_ratings(
            self, 
            submissions: List[MinerSubmission], 
            float_scores: List[float]
    ) -> None:
        """
        Update the ratings of the miners  based on their performance.
        """
        raw_scores = {}
        for fs, submission in zip(float_scores, submissions):
            raw_scores[submission.miner_hotkey] = fs

        sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)

        ratings_groups = []
        for k, v in self.ratings.items():
            if k in raw_scores:
                ratings_groups.append({k: v})

        ranks = []
        for x in ratings_groups:
            for mhk, _ in x.items():
                for i, (mhk2, _) in enumerate(sorted_scores):
                    if mhk == mhk2:
                        ranks.append(i)
                        break

        new_ratings = self.env.rate(ratings_groups, ranks=ranks)

        # Save new ratings
        for rating_result in new_ratings:
            for mhk, rating in rating_result.items():
                self.ratings[mhk] = rating