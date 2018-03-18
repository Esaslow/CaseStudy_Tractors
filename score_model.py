import sys
import numpy as np
import pandas as pd
from performotron import Comparer

class RMLSEComparer(Comparer):
    def score(self, predictions):
        log_diff = np.log(predictions+1) - np.log(self.target+1)
        return np.sqrt(np.mean(log_diff**2))
        
if __name__=='__main__':
    infile = sys.argv[1]
    predictions = pd.read_csv(infile)
    predictions.set_index('SalesID')
    test_solution = pd.read_csv('data/do_not_open/test_soln.csv')
    test_solution.set_index('SalesID')
    c = RMLSEComparer(test_solution.SalePrice)
    c.report_to_slack(predictions.SalePrice)
