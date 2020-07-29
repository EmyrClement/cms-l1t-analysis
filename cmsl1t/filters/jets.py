
def _pfJetID(jet):
    abs_eta = abs(jet.eta)
    isInnerJet = abs_eta <= 2.4
    isCentralJet = abs_eta <= 2.7
    isForwardCentralJet = (abs_eta > 2.7 and abs_eta <= 3.0)
    isForwardJet = abs_eta > 3.0
    reject_if = [
        jet.muMult != 0,
        isCentralJet and jet.nhef >= 0.9,
        isCentralJet and jet.nemef >= 0.9,
        isCentralJet and (jet.cMult + jet.nMult) <= 1,
        isCentralJet and jet.mef >= 0.8,
        isInnerJet and jet.chef <= 0,
        isInnerJet and jet.cMult <= 0,
        isInnerJet and jet.cemef >= 0.9,
        isForwardCentralJet and jet.nhef >= 0.98,
        isForwardCentralJet and jet.nemef <= 0.01,
        isForwardCentralJet and jet.nMult <= 2,
        isForwardJet and jet.nemef >= 0.9,
        isForwardJet and jet.nMult <= 10
    ]
    if any(reject_if):
        return False
    return True


def pfJetFilter(jets):
    return [jet for jet in jets if _pfJetID(jet)]

def _jetEtaFilter(jet, lowAbsEta, highAbsEta):
    abs_eta = abs(jet.eta)
    if abs_eta < lowAbsEta or abs_eta > highAbsEta:
        return False
    return True

def jetEtaFilter(jets, lowAbsEta, highAbsEta):
    return [jet for jet in jets if _jetEtaFilter(jet, lowAbsEta, highAbsEta)]