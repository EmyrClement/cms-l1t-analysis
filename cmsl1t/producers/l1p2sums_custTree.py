from __future__ import print_function

from cmsl1t.energySums import EnergySum, Met, Mht
from .base import BaseProducer


class Producer(BaseProducer):

    def __init__(self, inputs, outputs, **kwargs):
        self._expected_input_order = ['Ht', 'MET', 'MHT']
        super(Producer, self).__init__(inputs, outputs, **kwargs)

    def produce(self, event):
        setattr(event, self._outputs[0] + '_Met', Met(event[self._inputs[1]], 0.))
        setattr(event, self._outputs[0] + '_MetHF', Met(event['L1PhaseIPFJet_phaseIPFJetMETHF'], 0.))
        setattr(event, self._outputs[0] + '_Htt', EnergySum(event[self._inputs[0]]))
        setattr(event, self._outputs[0] + '_Mht', Mht(event[self._inputs[2]], 0.))

        return True
