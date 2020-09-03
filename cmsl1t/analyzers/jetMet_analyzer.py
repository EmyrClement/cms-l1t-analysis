import os
from cmsl1t.analyzers.BaseAnalyzer import BaseAnalyzer
from cmsl1t.plotting.efficiency import EfficiencyPlot
from cmsl1t.collections import EfficiencyCollection
from cmsl1t.plotting.onlineVsOffline import OnlineVsOffline
from cmsl1t.plotting.resolution import ResolutionPlot
from cmsl1t.plotting.resolution_vs_X import ResolutionVsXPlot
from cmsl1t.playground.jetfilters import pfJetFilter
# from cmsl1t.playground.metfilters import pfMetFilter
from cmsl1t.filters import pfMetFilter
from cmsl1t.filters import LuminosityFilter
import cmsl1t.recalc.met as recalc
from cmsl1t.energySums import EnergySum, Met
from math import pi
import pprint
from collections import namedtuple
import numpy as np
import csv
import cmsl1t
from cmsl1t.jet import match


def types(doEmu, doReco, doGen):

    sum_types = []
    jet_types = []
    if doReco:
        sum_types.extend(["caloHT", "pfHT", "caloMETHF",
                          "caloMETBE", "pfMET_NoMu", "caloMHT", "pfMHT"])
        jet_types.extend(["caloJetET_BE", "caloJetET_HF",
                          "pfJetET_BE", "pfJetET_HF"])
    if doGen:
        sum_types.extend(["genHT", "genMETHF", "genMETBE", "genMHT"])
        jet_types.extend(["genJetET_BE", "genJetET_HF", "genJetET_Barrel", "genJetET_EndcapWithTracks", "genJetET_EndcapNoTracks"])

    if doEmu:
        sum_types += [t + '_Emu' for t in sum_types]
        jet_types += [t + '_Emu' for t in jet_types]

    return sum_types, jet_types


# Eta ranges so we can put |\eta| < val as the legend header on the
# efficiency plots.
ETA_RANGES = dict(
    HT="|\\eta| < 2.4",
    METBE="|\\eta| < 3.0",
    METHF="|\\eta| < 5.0",
    JetET_BE="|\\eta| < 3.0",
    JetET_HF="3.0 < |\\eta| < 5.0",
    JetET_Barrel="|\\eta| < 1.5",
    JetET_EndcapWithTracks="1.5 < |\\eta| < 2.4",
    JetET_EndcapNoTracks="2.4 < |\\eta| < 3.0",
)

ALL_THRESHOLDS = dict(
    HT=[120, 200, 320],
    METBE=[80, 100, 120],
    METHF=[80, 100, 120],
    JetET=[35, 90, 120],
    MHT=[80, 100, 120]
)


def extractSums(event, doEmu, doReco, doGen):
    offline = dict()
    online = dict()
    if doReco:
        offline.update(dict(
            caloHT=EnergySum(event.Sums_caloHt),
            pfHT=EnergySum(event.Sums_Ht),
            caloMETBE=Met(event.Sums_caloMetBE, event.Sums_caloMetPhiBE),
            caloMETHF=Met(event.Sums_caloMet, event.Sums_caloMetPhi),
            pfMET_NoMu=Met(event.Sums_pfMetNoMu, event.Sums_pfMetNoMuPhi),
            caloMHT=EnergySum(event.Sums_caloMht),
            pfMHT=EnergySum(event.Sums_Mht)
        ))
        online.update(dict(
            caloHT=event.l1Sums_Htt,
            pfHT=event.l1Sums_Htt,
            caloMETBE=event.l1Sums_Met,
            caloMETHF=event.l1Sums_MetHF,
            pfMET_NoMu=event.l1Sums_MetHF,
            caloMHT=event.l1Sums_Mht,
            pfMHT=event.l1Sums_Mht,

        ))
        if doEmu:
            offline.update(dict(
                caloHT_Emu=EnergySum(event.Sums_caloHt),
                pfHT_Emu=EnergySum(event.Sums_Ht),
                caloMETBE_Emu=Met(event.Sums_caloMetBE,
                                  event.Sums_caloMetPhiBE),
                caloMETHF_Emu=Met(event.Sums_caloMet, event.Sums_caloMetPhi),
                pfMET_NoMu_Emu=Met(event.Sums_pfMetNoMu,
                                   event.Sums_pfMetNoMuPhi),
                caloMHT_Emu=EnergySum(event.Sums_caloMht),
                pfMHT_Emu=EnergySum(event.Sums_Mht),

            ))
            online.update(dict(
                caloHT_Emu=event.l1EmuSums_Htt,
                pfHT_Emu=event.l1EmuSums_Htt,
                caloMETBE_Emu=event.l1EmuSums_Met,
                caloMETHF_Emu=event.l1EmuSums_MetHF,
                pfMET_NoMu_Emu=event.l1EmuSums_MetHF,
                caloMHT_Emu=event.l1EmuSums_Mht,
                pfMHT_Emu=event.l1EmuSums_Mht,
            ))

    if doGen:
        offline.update(dict(
            genHT=event.genSums_HT,
            genMETHF=event.genSums_MetHF,
            genMETBE=event.genSums_MetBE,
            genMHT=event.genSums_Mht,
        ))
        online.update(dict(
            genHT=event.l1Sums_Htt,
            genMETHF=event.l1Sums_MetHF,
            genMETBE=event.l1Sums_Met,
            genMHT=event.l1Sums_Mht,
        ))
        if doEmu:
            offline.update(dict(
                genHT_Emu=event.genSums_HT,
                genMETHF_Emu=event.genSums_MetHF,
                genMETBE_Emu=event.genSums_MetBE,
                genMHT=event.genSums_Mht,
            ))
            online.update(dict(
                genHT_Emu=event.l1EmuSums_Htt,
                genMETHF_Emu=event.l1EmuSums_MetHF,
                genMETBE_Emu=event.l1EmuSums_Met,
                genMHT_Emu=event.l1EmuSums_Mht,
            ))
    return offline, online


class Analyzer(BaseAnalyzer):

    def __init__(self, **kwargs):
        super(Analyzer, self).__init__(**kwargs)

        lumiMuDict = dict()
        run_lumi_csv = os.path.join(cmsl1t.PROJECT_ROOT, 'run_lumi.csv')
        with open(run_lumi_csv, 'rb') as runLumiFile:
            reader = csv.reader(runLumiFile, delimiter=',')
            for line in reader:
                lumiMuDict[(int(line[1]), int(line[2]))] = float(line[3])
        self._lumiMu = lumiMuDict

        self._lumiFilter = None
        self._lumiJson = self.params['lumiJson']
        if self._lumiJson:
            self._lumiFilter = LuminosityFilter(self._lumiJson)

        self._lastRunAndLumi = (-1, -1)
        self._processLumi = True

        # TODO: this needs changing, these should be analyser parameters
        # or even move out into separate calls of the same analyzer
        loaded_trees = self.params['load_trees']
        self._doVertex = 'recoTree' in loaded_trees
        self._doEmu = 'emuUpgrade' in loaded_trees or 'p2UpgradeEmu' in loaded_trees
        self._doReco = 'recoTree' in loaded_trees
        self._doGen = 'genTree' in loaded_trees or 'p2Upgrade' in loaded_trees
        self._doPhase2 = 'p2Upgrade' in loaded_trees

        self._sumTypes, self._jetTypes = types(
            self._doEmu, self._doReco, self._doGen)

        if 'whichJet' in self.params.keys():
            self._whichJet = self.params['whichJet']
        else : 
            self._whichJet = 0

        for name in self._sumTypes:
            eff_plot = EfficiencyPlot("L1", "offline_" + name)
            res_plot = ResolutionPlot("energy", "L1", "offline_" + name)
            twoD_plot = OnlineVsOffline("L1", "offline_" + name)
            self.register_plotter(eff_plot)
            self.register_plotter(res_plot)
            self.register_plotter(twoD_plot)
            setattr(self, name + "_eff", eff_plot)
            setattr(self, name + "_res", res_plot)
            setattr(self, name + "_2D", twoD_plot)

            eff_plot_HR = EfficiencyPlot("L1", "offline_" + name + "_HiRange")
            twoD_plot_HR = OnlineVsOffline(
                "L1", "offline_" + name + "_HiRange")
            self.register_plotter(eff_plot_HR)
            self.register_plotter(twoD_plot_HR)
            setattr(self, name + "_eff_HR", eff_plot_HR)
            setattr(self, name + "_2D_HR", twoD_plot_HR)

        for angle in self._sumTypes:
            name = angle + "_phi"
            if 'HT' in angle:
                continue
            res_plot = ResolutionPlot("phi", "L1", "offline_" + name)
            twoD_plot = OnlineVsOffline("L1", "offline_" + name)
            self.register_plotter(res_plot)
            self.register_plotter(twoD_plot)
            setattr(self, name + "_res", res_plot)
            setattr(self, name + "_2D", twoD_plot)

        for name in self._jetTypes:
            eff_plot = EfficiencyPlot("L1", "offline_" + name)
            res_plot = ResolutionPlot("energy", "L1", "offline_" + name)
            resPhi_plot = ResolutionPlot("phi", "L1", "offline_" + name + "Phi")
            resEta_plot = ResolutionPlot("eta", "L1", "offline_" + name + "Eta")
            twoD_plot = OnlineVsOffline("L1", "offline_" + name)
            self.register_plotter(eff_plot)
            self.register_plotter(res_plot)
            self.register_plotter(resPhi_plot)
            self.register_plotter(resEta_plot)
            self.register_plotter(twoD_plot)
            setattr(self, name + "_eff", eff_plot)
            setattr(self, name + "_res", res_plot)
            setattr(self, name + "_resPhi", resPhi_plot)
            setattr(self, name + "_resEta", resEta_plot)
            setattr(self, name + "_2D", twoD_plot)

            eff_plot_HR = EfficiencyPlot("L1", "offline_" + name + "_HiRange")
            twoD_plot_HR = OnlineVsOffline(
                "L1", "offline_" + name + "_HiRange")
            self.register_plotter(eff_plot_HR)
            self.register_plotter(twoD_plot_HR)
            setattr(self, name + "_eff_HR", eff_plot_HR)
            setattr(self, name + "_2D_HR", twoD_plot_HR)

        self.res_vs_eta_CentralJets = ResolutionVsXPlot(
            "energy", "", "", "pfJet_eta")
        self.register_plotter(self.res_vs_eta_CentralJets)
        self.res_vs_eta_CentralGenJets = ResolutionVsXPlot(
            "energy", "", "", "genJet_eta")
        self.register_plotter(self.res_vs_eta_CentralGenJets)

    def prepare_for_events(self, reader):
        puBins = self.params['pu_bins']
        puBins_HR = [0, 999]

        Config = namedtuple(
            "Config",
            "name off_title on_title min max",
        )
        cfgs = []
        if self._doReco:
            cfgs.extend([
                Config("caloHT", "Offline Calo HT", "L1 HT", 30, 830),
                Config("pfHT", "Offline PF HT", "L1 HT", 30, 830),
                Config("caloMETHF", "Offline Calo MET HF", "L1 MET HF", 0, 400),
                Config("caloMETBE", "Offline Calo MET BE", "L1 MET BE", 0, 400),
                Config("pfMET_NoMu", "Offline PF MET NoMu",
                       "L1 MET HF", 0, 400),
                Config("caloJetET_BE", "Offline Central Calo Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("caloJetET_HF", "Offline Forward Calo Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("pfJetET_BE", "Offline Central PF Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("pfJetET_HF", "Offline Forward PF Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("caloMHT", "Offline Calo MHT", "L1 MHT", 0, 400),
                Config("pfMHT", "Offline PF MHT", "L1 MHT", 0, 400),
            ])
        if self._doGen:
            cfgs.extend([
                Config("genJetET_BE", "Central Gen Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("genJetET_HF", "Forward Gen Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("genJetET_Barrel", "Barrel Gen Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("genJetET_EndcapWithTracks", "Endcap (with tracks) Gen Jet ET",
                       "L1 Jet ET", 20, 420),
                Config("genJetET_EndcapNoTracks", "Endcap (no tracks) Gen Jet ET",
                       "L1 Jet ET", 20, 420),
            ])
            cfgs.extend([
                Config("genHT", "Gen HT", "L1 HT", 30, 830),
                Config("genMETHF", "Gen MET HF", "L1 MET HF", 0, 400),
                Config("genMETBE", "Gen MET BE", "L1 MET BE", 0, 400),
                Config("genMHT", "Gen MHT", "L1 MHT", 0, 400),
            ])

        self._plots_from_cfgs(cfgs, puBins)
        if self._doEmu:
            self._plots_from_cfgs(cfgs, puBins, emulator=True)
        self._plots_from_cfgs(cfgs, puBins_HR, high_range=True)
        if self._doEmu:
            self._plots_from_cfgs(
                cfgs, puBins_HR, emulator=True, high_range=True)

        self.res_vs_eta_CentralJets.build(
            "Online Jet energy (GeV)",
            "Offline Jet energy (GeV)",
            "Offline Jet Eta (rad)",
            puBins,
            50, -0.5, 3.5, 50, -5.0, 5.0,
        )
        self.res_vs_eta_CentralGenJets.build(
            "Online Jet energy (GeV)",
            "Offline Jet energy (GeV)",
            "Offline Jet Eta (rad)",
            puBins,
            50, -0.5, 3.5, 50, -5.0, 5.0,
        )
        return True

    def _plots_from_cfgs(self, cfgs, puBins, emulator=False, high_range=False):
        suffix = ""
        prefix = ""
        if high_range:
            suffix = '_HR'
        if emulator:
            prefix = '_Emu'

        if 'thresholds' in self.params:
            allThresholds = self.params['thresholds']
        else:
            allThresholds = ALL_THRESHOLDS

        for cfg in cfgs:
            eff_plot = getattr(self, cfg.name + prefix + "_eff" + suffix)
            twoD_plot = getattr(self, cfg.name + prefix + "_2D" + suffix)

            thresholds = []
            for l1trig, thresh in allThresholds.items():
                if emulator and "Emu" not in l1trig or not emulator and "Emu" in l1trig:
                    continue
                if l1trig.replace("_Emu", "") in cfg.name:
                    thresholds = thresh
                    break
            if 'pfMET' in cfg.name:
                if emulator:
                    thresholds = allThresholds.get("METHF_Emu")
                else:
                    thresholds = allThresholds.get("METHF")

            etaRange = ""
            for l1trig, etaRange in ETA_RANGES.items():
                if l1trig in cfg.name:
                    etaRange = etaRange
                    break
            if 'pfMET' in cfg.name:
                etaRange = ETA_RANGES.get("METHF")

            params = [
                cfg.on_title, cfg.off_title + " (GeV)", puBins, thresholds,
                (cfg.max - cfg.min) / 10, cfg.min, cfg.max,
            ]
            if high_range:
                params = [
                    cfg.on_title, cfg.off_title + " (GeV)", puBins, thresholds,
                    105, 0, 2100,
                ]
                if "HT" in cfg.name:
                    params = [
                        cfg.on_title, cfg.off_title + " (GeV)",
                        puBins, thresholds, 105, 30, 2130
                    ]
                if "JetET" in cfg.name:
                    params = [
                        cfg.on_title, cfg.off_title + " (GeV)",
                        puBins, thresholds, 105, 20, 2120
                    ]
                if "MET" in cfg.name:
                    params = [
                        cfg.on_title, cfg.off_title + " (GeV)",
                        puBins, thresholds, 100, 0, 2000
                    ]
                if "MHT" in cfg.name:
                    params = [
                        cfg.on_title, cfg.off_title + " (GeV)",
                        puBins, thresholds, 100, 0, 2000
                    ]

            eff_plot.build(*params, legend_title=etaRange)
            params.remove(thresholds)
            twoD_plot.build(*params)

            if high_range:
                continue
            res_plot = getattr(self, cfg.name + prefix + "_res" + suffix)
            res_plot.build(cfg.on_title, cfg.off_title,
                           puBins, 150, -3, 3, legend_title=etaRange)

            if 'JetET' in cfg.name:
                resPhi_plot = getattr(self, cfg.name + prefix + "_resPhi" + suffix)
                resPhi_plot.build(cfg.on_title + 'Phi', cfg.off_title + 'Phi',
                               puBins, 150, -0.5, 0.5, legend_title=etaRange)

                resEta_plot = getattr(self, cfg.name + prefix + "_resEta" + suffix)
                resEta_plot.build(cfg.on_title + 'Eta', cfg.off_title + 'Eta',
                               puBins, 150, -0.5, 0.5, legend_title=etaRange)

            if not hasattr(self, cfg.name + prefix + "_phi_res"):
                continue
            res_plot = getattr(self, cfg.name + prefix + "_phi_res")
            twoD_plot = getattr(self, cfg.name + prefix + "_phi_2D")
            twoD_plot.build(
                cfg.on_title + " Phi (rad)",
                cfg.off_title + " Phi (rad)",
                puBins, 100,
                -pi,
                2 * pi,
            )
            res_plot.build(
                cfg.on_title + " Phi",
                cfg.off_title + " Phi",
                puBins,
                100,
                -2 * pi,
                2 * pi,
                legend_title=etaRange,
            )

    def fill_histograms(self, entry, event):

        recoNVtx = 1
        genNVtx = 1

        if self._doReco or self._doVertex:
            recoNVtx = event.Vertex_nVtx
        if self._doGen and not self._doPhase2:
            genNVtx = event.Generator_nVtx

        if not self._doGen:
            if (event['run'], event['lumi']) in self._lumiMu:
                pileup = self._lumiMu[(event['run'], event['lumi'])]
            if not self._passesLumiFilter(event.run, event.lumi):
                return True

        offline, online = extractSums(
            event, self._doEmu, self._doReco, self._doGen)

        for name in self._sumTypes:
            if 'pfMET' in name and not pfMetFilter(event):
                continue
            on = online[name]
            off = offline[name]
            for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                if '_res' in suffix and (on.et < 0.01 or off.et < 30):
                    continue
                pileup = recoNVtx
                if 'gen' in name:
                    pileup = genNVtx

                getattr(self, name + suffix).fill(pileup, off.et, on.et)
            if hasattr(self, name + "_phi_res"):
                getattr(self, name + "_phi_res").fill(pileup, off.phi, on.phi)
                getattr(self, name + "_phi_2D").fill(pileup, off.phi, on.phi)

        if self._doReco and self._doEmu:
            goodRefJets = event.goodPFJets

            for refJet in goodRefJets:
                l1Jet = match(refJet, event.l1EmuJets)
                if not l1Jet:
                    continue
                if refJet.etCorr > 30.:
                    self.res_vs_eta_CentralJets.fill(
                        recoNVtx, refJet.eta, refJet.etCorr, l1Jet.et)

        if self._doGen and self._doEmu:
            goodRefJets = event.goodGenJets

            for refJet in goodRefJets:
                l1Jet = match(refJet, event.l1EmuJets)
                if not l1Jet:
                    continue
                if refJet.etCorr > 30.:
                    self.res_vs_eta_CentralGenJets.fill(
                        genNVtx, refJet.eta, refJet.etCorr, l1Jet.et)

        if self._doGen:
            leadingGenJet = None
            if event.goodGenJets:
                if len(event.goodGenJets) >= self._whichJet + 1: 
                    leadingGenJet = event.goodGenJets[self._whichJet]

            if leadingGenJet:

                genFillRegions = []
                if abs(leadingGenJet.eta) < 3.0:
                    genFillRegions = ['BE']
                    if abs(leadingGenJet.eta) < 1.5 : 
                        genFillRegions.append('Barrel')
                    elif abs(leadingGenJet.eta) < 3.0 :
                        # genFillRegions.append('Endcap')
                        if abs(leadingGenJet.eta) < 2.4 :
                            genFillRegions.append('EndcapWithTracks')
                        elif abs(leadingGenJet.eta) < 3.0 :
                            genFillRegions.append('EndcapNoTracks')
                else:
                    genFillRegions = ['HF']

                if self._doEmu:
                    genL1EmuJet = match(leadingGenJet, event.l1EmuJets)

                    # Only keep events where all gen jets up to self._whichJet are found
                    if self._whichJet > 0 :
                        for i in range(self._whichJet+1):
                            harderEmuJetMatch = match( event.goodGenJets[i], event.l1EmuJets )
                            if not harderEmuJetMatch:
                                genL1EmuJet = None

                    genL1EmuJetEt = None
                    genL1EmuJetPhi = None
                    genL1EmuJetEta = None
                    if genL1EmuJet:
                        genL1EmuJetEt = genL1EmuJet.et
                        genL1EmuJetPhi = genL1EmuJet.phi
                        genL1EmuJetEta = genL1EmuJet.eta
                    else:
                        genL1EmuJetEt = 0.
                        genL1EmuJetPhi = None
                        genL1EmuJetEta = None

                    for region in genFillRegions:
                        for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                            if '_res' in suffix and (genL1EmuJetEt == 0 or leadingGenJet.etCorr < 30):
                                continue
                            name = 'genJetET_{0}_Emu{1}'.format(region, suffix)
                            getattr(self, name).fill(
                                genNVtx, leadingGenJet.etCorr, genL1EmuJetEt,
                            )
                        for suffix in ['_resPhi', '_resEta']:
                            if '_res' in suffix and (genL1EmuJetEt == 0 or leadingGenJet.etCorr < 30):
                                continue
                            name = 'genJetET_{0}_Emu{1}'.format(region, suffix)
                            if 'Phi' in suffix:
                                getattr(self, name).fill(
                                    genNVtx, leadingGenJet.phi, genL1EmuJetPhi,
                                )
                            elif 'Eta' in suffix:
                                getattr(self, name).fill(
                                    genNVtx, leadingGenJet.eta, genL1EmuJetEta,
                                )

                genL1Jet = match(leadingGenJet, event.l1Jets)

                genL1JetEt = None
                genL1JetPhi = None
                genL1JetEta = None
                if genL1Jet:
                    genL1JetEt = genL1Jet.et
                    genL1JetPhi = genL1Jet.phi
                    genL1JetEta = genL1Jet.eta
                else:
                    genL1JetEt = 0.
                    genL1JetPhi = None
                    genL1JetEta = None
                    # if ( leadingGenJet.et > 500 ) :
                    #     print ("Didn't find a matching gen jet : ", leadingGenJet.et)

                for region in genFillRegions:
                    for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                        if '_res' in suffix and (genL1JetEt == 0 or leadingGenJet.etCorr < 30):
                            continue
                        name = 'genJetET_{0}{1}'.format(region, suffix)
                        getattr(self, name).fill(
                            genNVtx, leadingGenJet.etCorr, genL1JetEt,
                        )

                    for suffix in ['_resPhi', '_resEta']:
                        if '_res' in suffix and (genL1JetEt == 0 or leadingGenJet.etCorr < 30):
                            continue
                        name = 'genJetET_{0}{1}'.format(region, suffix)
                        if 'Phi' in suffix:
                            getattr(self, name).fill(
                                genNVtx, leadingGenJet.phi, genL1JetPhi,
                            )
                        elif 'Eta' in suffix:
                            getattr(self, name).fill(
                                genNVtx, leadingGenJet.eta, genL1JetEta,
                            )


        if self._doReco:
            leadingPFJet, leadingCaloJet = None, None
            if event.goodPFJets:
                if len(event.goodPFJets) >= self._whichJet + 1: leadingPFJet = event.goodPFJets[self._whichJet]
            if event.caloJets:
                if len(event.caloJets) >= self._whichJet + 1: leadingCaloJet = event.caloJets[self._whichJet]

            if leadingPFJet and leadingPFJet.etCorr > 20:
                pfFillRegions = []
                if abs(leadingPFJet.eta) < 3.0:
                    pfFillRegions = ['BE']
                    if abs(leadingPFJet.eta) < 1.5 : 
                        pfFillRegions.append('Barrel')
                    elif abs(leadingPFJet.eta) < 3.0 :
                        # pfFillRegions.append('Endcap')
                        if abs(leadingPFJet.eta) < 2.4 :
                            pfFillRegions.append('EndcapWithTracks')
                        elif abs(leadingPFJet.eta) < 3.0 :
                            pfFillRegions.append('EndcapNoTracks')
                else:
                    pfFillRegions = ['HF']

                if self._doEmu:
                    pfL1EmuJet = match(leadingPFJet, event.l1EmuJets)
                    if pfL1EmuJet:
                        pfL1EmuJetEt = pfL1EmuJet.et
                    else:
                        pfL1EmuJetEt = 0.

                    for region in pfFillRegions:
                        for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                            if '_res' in suffix and (pfL1EmuJetEt == 0 or leadingPFJet.etCorr < 30):
                                continue
                            name = 'pfJetET_{0}_Emu{1}'.format(region, suffix)
                            getattr(self, name).fill(
                                recoNVtx, leadingPFJet.etCorr, pfL1EmuJetEt,
                            )

                pfL1Jet = match(leadingPFJet, event.l1Jets)
                if pfL1Jet:
                    pfL1JetEt = pfL1Jet.et
                else:
                    pfL1JetEt = 0.

                for region in pfFillRegions:
                    for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                        if '_res' in suffix and (pfL1JetEt == 0 or leadingPFJet.etCorr < 30):
                            continue
                        name = 'pfJetET_{0}{1}'.format(region, suffix)
                        getattr(self, name).fill(
                            recoNVtx, leadingPFJet.etCorr, pfL1JetEt,
                        )

            if leadingCaloJet and leadingCaloJet.etCorr > 20:

                caloFillRegions = []
                if abs(leadingCaloJet.eta) < 3.0:
                    caloFillRegions = ['BE']
                    if abs(leadingCaloJet.eta) < 1.5 : 
                        caloFillRegions.append('Barrel')
                    elif abs(leadingCaloJet.eta) < 3.0 :
                        caloFillRegions.append('Endcap')
                        if abs(leadingCaloJet.eta) < 2.4 :
                            caloFillRegions.append('EndcapWithTracks')
                        elif abs(leadingCaloJet.eta) < 3.0 :
                            caloFillRegions.append('EndcapNoTracks')
                else:
                    caloFillRegions = ['HF']

                if self._doEmu:
                    caloL1EmuJet = match(leadingCaloJet, event.l1EmuJets)
                    if caloL1EmuJet:
                        caloL1EmuJetEt = caloL1EmuJet.et
                    else:
                        caloL1EmuJetEt = 0.

                    for region in caloFillRegions:
                        for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                            if '_res' in suffix and (caloL1EmuJetEt == 0 or leadingCaloJet.etCorr < 30):
                                continue
                            name = 'caloJetET_{0}_Emu{1}'.format(
                                region, suffix)
                            getattr(self, name).fill(
                                recoNVtx, leadingCaloJet.etCorr, caloL1EmuJetEt,
                            )

                caloL1Jet = match(leadingCaloJet, event.l1Jets)
                if caloL1Jet:
                    caloL1JetEt = caloL1Jet.et
                else:
                    caloL1JetEt = 0.

                for region in caloFillRegions:
                    for suffix in ['_eff', '_res', '_2D', '_eff_HR', '_2D_HR']:
                        if '_res' in suffix and (caloL1JetEt == 0 or leadingCaloJet.etCorr < 30):
                            continue
                        name = 'caloJetET_{0}{1}'.format(region, suffix)
                        getattr(self, name).fill(
                            recoNVtx, leadingCaloJet.etCorr, caloL1JetEt,
                        )

        return True

    def _passesLumiFilter(self, run, lumi):
        if self._lumiFilter is None:
            return True
        if (run, lumi) == self._lastRunAndLumi:
            return self._processLumi

        self._lastRunAndLumi = (run, lumi)
        self._processLumi = self._lumiFilter(run, lumi)

        return self._processLumi

    def make_plots(self, other_analyzers=None):
        """
        Custom version, does what the normal one does but also overlays whatever you like.
        """

        if self._doReco:

            plot_names = [
                'caloHT', 'pfHT', 'caloMETBE', 'caloMETHF',
                'pfMET_NoMu', 'caloJetET_BE', 'caloJetET_HF',
                'pfJetET_BE', 'pfJetET_HF', 'caloMHT', 'pfMHT'
            ]

        if self._doGen:

            plot_names = ['genJetET_BE', 'genJetET_HF', 'genJetET_Barrel', 'genJetET_EndcapWithTracks', 'genJetET_EndcapNoTracks', 'genHT', 'genMETBE', 'genMETHF', 'genMHT',]

        for plot_name in plot_names:
            getattr(self, plot_name + '_eff').draw()
            getattr(self, plot_name + '_eff_HR').draw()
            getattr(self, plot_name + '_res').draw()
            if 'JetET' in plot_name:
                getattr(self, plot_name + '_resPhi').draw()
                getattr(self, plot_name + '_resEta').draw()

        if self._doEmu:

            for plot_name in plot_names:
                getattr(self, plot_name + '_eff').overlay([
                    getattr(self, plot_name + '_Emu_eff')])
                getattr(self, plot_name + '_res').overlay([
                    getattr(self, plot_name + '_Emu_res')])

        if self._doComp:

            for plot_name in plot_names:
                getattr(self, plot_name + '_eff').overlay(
                    [getattr(other_analyzer, plot_name + '_eff')
                        for other_analyzer in other_analyzers])
                getattr(self, plot_name + '_res').overlay(
                    [getattr(other_analyzer, plot_name + '_res')
                        for other_analyzer in other_analyzers])
                if 'JetET' in plot_name:
                    getattr(self, plot_name + '_resPhi').overlay(
                        [getattr(other_analyzer, plot_name + '_resPhi')
                            for other_analyzer in other_analyzers])
                    getattr(self, plot_name + '_resEta').overlay(
                        [getattr(other_analyzer, plot_name + '_resEta')
                            for other_analyzer in other_analyzers])

        return True
