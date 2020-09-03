from __future__ import division
from cmsl1t.energySums import EnergySum, Met, Mht
from .base import BaseProducer


class Producer(BaseProducer):

    def __init__(self, inputs, outputs, **kwargs):
        self._expected_input_order = ['genJetHt', 'genMet', 'genJetMHT']
        super(Producer, self).__init__(inputs, outputs, **kwargs)

    def produce(self, event):
        setattr(event, self._outputs[0] + '_MetBE', Met(event['L1PhaseIPFJet_genMet'], 0.))
        setattr(event, self._outputs[0] + '_MetHF', Met(event['L1PhaseIPFJet_genMet'], 0.))
        setattr(event, self._outputs[0] + '_HT', EnergySum(event['L1PhaseIPFJet_genJetHt']))
        setattr(event, self._outputs[0] + '_Mht', Mht(event['L1PhaseIPFJet_genJetMHT'], 0.))

        return True
