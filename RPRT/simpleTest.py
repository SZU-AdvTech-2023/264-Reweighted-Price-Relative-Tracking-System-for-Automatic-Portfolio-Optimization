from universal import tools
from universal.algos import CRP, RMR, OLMAR, BNN, CWMR
from universal.algos import BAH
from universal.algos.olu import OLU
from universal.algos.gpolu import GPOLU
from universal.tools import dataset
from universal.algos.rprt import RPRT
from universal.algos.nrprt import NRPRT

if __name__ == '__main__':
  # Run CRP on a computed-generated portfolio of 3 stocks and plot the results
  tools.quickrun(NRPRT(),dataset('sp500'))