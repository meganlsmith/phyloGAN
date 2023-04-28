"""Parts of this code are adapted from Wang et al., 2021, Molecular Ecology Resources, https://doi.org/10.1111/1755-0998.13386."""
from scipy.stats import norm

class Proposer(object):

    def proposal(self, curr_value, multiplier, proposal_width, proposal_min, proposal_max):

        if multiplier <= 0: # last iter
            return curr_value
    
        # normal around current value (make sure we don't go outside bounds)
        new_value = norm(curr_value, proposal_width*multiplier).rvs()
        new_value = self.fit_to_range(new_value, proposal_min, proposal_max)
        # if the parameter hits the min or max it tends to get stuck
        if new_value == curr_value or new_value == proposal_min or new_value == \
            proposal_max:
            return self.proposal(curr_value, multiplier, proposal_width, proposal_min, proposal_max) # recurse
        else:
            return new_value
    
    def fit_to_range(self, value, proposal_min, proposal_max):
        value = min(value, proposal_max)
        return max(value, proposal_min)
