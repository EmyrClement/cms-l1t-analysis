"""
Study the MET distibutions and various PUS schemes
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import ROOT
from tabulate import tabulate
import os
import csv
import cmsl1t
from cmsl1t.analyzers.BaseAnalyzer import BaseAnalyzer
from cmsl1t.plotting.rates import RatesPlot
from cmsl1t.plotting.rate_vs_pileup import RateVsPileupPlot
from cmsl1t.filters import LuminosityFilter
import cmsl1t.hist.binning as bn
from cmsl1t.utils.hist import cumulative_hist
from cmsl1t.utils.hist import normalise_to_collision_rate


def types(doEmu):

    sum_types = ["HT", "METBE", "METHF"]
    jet_types = ["JetET_BE", "JetET_HF", "JetET_Barrel", "JetET_EndcapWithTracks", "JetET_EndcapNoTracks"]

    if doEmu:
        sum_types += [t + '_Emu' for t in sum_types]
        jet_types += [t + '_Emu' for t in jet_types]

    return sum_types, jet_types


def extractSums(event, doEmu):
    online = dict(
        HT=event.l1Sums_Htt,
        METBE=event.l1Sums_Met,
        METHF=event.l1Sums_MetHF
    )
    if doEmu:
        online['HT_Emu'] = event.l1EmuSums_Htt
        online['METBE_Emu'] = event.l1EmuSums_Met
        online['METHF_Emu'] = event.l1EmuSums_MetHF

    return online


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
    HT_Emu="|\\eta| < 2.4",
    METBE_Emu="|\\eta| < 3.0",
    METHF_Emu="|\\eta| < 5.0",
    JetET_BE_Emu="|\\eta| < 3.0",
    JetET_HF_Emu="3.0 < |\\eta| < 5.0",
    JetET_Barrel_Emu="|\\eta| < 1.5",
    JetET_EndcapWithTracks_Emu="1.5 < |\\eta| < 2.4",
    JetET_EndcapNoTracks_Emu="2.4 < |\\eta| < 3.0",
)


class Analyzer(BaseAnalyzer):

    def __init__(self, **kwargs):
        super(Analyzer, self).__init__(**kwargs)
        self.triggerName = self.params['triggerName']
        self.thresholds = self.params['thresholds']
        self.puBins = self.params['pu_bins']
        self.menuRates = self.params['rates']

        loaded_trees = self.params['load_trees']
        self._doGen = 'genTree' in loaded_trees or 'p2Upgrade' in loaded_trees
        self._doEmu = 'emuUpgrade' in loaded_trees or 'p2UpgradeEmu' in loaded_trees

        lumiMuDict = dict()
        with open(os.path.join(cmsl1t.PROJECT_ROOT, 'run_lumi.csv'), 'rb') as runLumiFile:
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
        self._sumTypes, self._jetTypes = types(self._doEmu)

        if 'whichJet' in self.params.keys():
            self._whichJet = self.params['whichJet']
        else : 
            self._whichJet = 0

        for name in self._sumTypes + self._jetTypes:
            rates_plot = RatesPlot(name)
            self.register_plotter(rates_plot)
            setattr(self, name + "_rates", rates_plot)

            rate_vs_pileup_plot = RateVsPileupPlot(name)
            self.register_plotter(rate_vs_pileup_plot)
            setattr(self, name + "_rate_vs_pileup", rate_vs_pileup_plot)

    def prepare_for_events(self, reader):
        # bins = np.arange(0.0, 400.0, 1.0)
        puBins = self.puBins
        thresholds = self.thresholds

        for name in self._sumTypes + self._jetTypes:
            trig_thresholds = thresholds.get(name)
            if(trig_thresholds is None):
                if "Emu" in name:
                    trig_thresholds = thresholds.get(name.replace('_Emu', ''))
                else:
                    print(
                        'Error: Please specify thresholds in the config .yaml in dictionary format')

            rates_plot = getattr(self, name + "_rates")
            rates_plot.build("L1 " + name, puBins, 400, 0, 400, ETA_RANGES.get(name))

            rate_vs_pileup_plot = getattr(self, name + "_rate_vs_pileup")
            rate_vs_pileup_plot.build("L1 " + name, trig_thresholds, 18, 20, 56, ETA_RANGES.get(name))

        '''
        self.rates = HistogramsByPileUpCollection(
            pileupBins=puBins, dimensions=2)
        for thing in object_types:
            self.rates.add(thing, bins)
        '''

        return True

    '''
    def reload_histograms(self, input_file):
        # Something like this needs to be implemented still
        self.rates = HistogramsByPileUpCollection.from_root(input_file)
        return True
    '''

    def fill_histograms(self, entry, event):
        # Get pileup if ntuples have reco trees in them.
        # If not, set PU to 1 so that it fills the (only) pu bin.

        try:
            pileup = event.nVertex
        except AttributeError:
            pileup = 1.

        if not self._doGen:
            if not self._passesLumiFilter(event['run'], event['lumi']):
                return True
            if (event['run'], event['lumi']) in self._lumiMu:
                pileup = self._lumiMu[(event['run'], event['lumi'])]

        # Sums:
        online = extractSums(event, self._doEmu)
        for name in self._sumTypes:
            on = online[name]
            getattr(self, name + "_rates").fill(pileup, on.et)
            getattr(self, name + "_rate_vs_pileup").fill(pileup, on.et)

        # All jets:
        l1JetEts = [jet.et for jet in event.l1Jets]
        nJets = len(l1JetEts)
        if nJets > 0:
            if ( nJets > self._whichJet ):
                maxL1JetEt = sorted(l1JetEts)[self._whichJet]
            else:
                maxL1JetEt = 0
        else:
            maxL1JetEt = 0.

        if self._doEmu:
            l1EmuJetEts = [jet.et for jet in event.l1EmuJets]
            nEmuJets = len(l1EmuJetEts)
            if nEmuJets > 0:
                maxL1EmuJetEt = max(l1EmuJetEts)
            else:
                maxL1EmuJetEt = 0.

        # Central Jets:
        l1BEJets = [jet for jet in event.l1Jets if abs(jet.eta) < 3.0]
        l1BEJetEts = [beJet.et for beJet in l1BEJets]
        nBEJets = len(l1BEJets)
        # if nBEJets > 0:
        #     maxL1BEJetEt = max(l1BEJetEts)
        # else:
        #     maxL1BEJetEt = 0.
        if nBEJets > 0:
            if ( nBEJets > self._whichJet ):
                maxL1BEJetEt = sorted(l1BEJetEts, reverse=True)[self._whichJet]
            else:
                maxL1BEJetEt = 0
        else:
            maxL1BEJetEt = 0.


        if self._doEmu:
            l1EmuBEJets = [jet for jet in event.l1EmuJets if abs(jet.eta) < 3.0]
            l1EmuBEJetEts = [beJet.et for beJet in l1EmuBEJets]
            nEmuBEJets = len(l1EmuBEJetEts)
            if nEmuBEJets > 0:
                maxL1EmuBEJetEt = max(l1EmuBEJetEts)
            else:
                maxL1EmuBEJetEt = 0.

        # Barrel Jets:
        l1BarrelJets = [jet for jet in event.l1Jets if abs(jet.eta) < 1.5]
        l1BarrelJetEts = [BarrelJet.et for BarrelJet in l1BarrelJets]
        nBarrelJets = len(l1BarrelJets)
        if nBarrelJets > 0:
            maxL1BarrelJetEt = max(l1BarrelJetEts)
        else:
            maxL1BarrelJetEt = 0.

        if self._doEmu:
            l1EmuBarrelJets = [jet for jet in event.l1EmuJets if abs(jet.eta) < 1.5]
            l1EmuBarrelJetEts = [BarrelJet.et for BarrelJet in l1EmuBarrelJets]
            nEmuBarrelJets = len(l1EmuBarrelJetEts)
            if nEmuBarrelJets > 0:
                maxL1EmuBarrelJetEt = max(l1EmuBarrelJetEts)
            else:
                maxL1EmuBarrelJetEt = 0.

        # Endcap with tracks Jets:
        l1EndcapWithTracksJets = [jet for jet in event.l1Jets if ( abs(jet.eta) > 1.5 and abs(jet.eta) < 2.4 ) ]
        l1EndcapWithTracksJetEts = [EndcapJet.et for EndcapJet in l1EndcapWithTracksJets]
        nEndcapWithTracksJets = len(l1EndcapWithTracksJets)
        if nEndcapWithTracksJets > 0:
            maxL1EndcapWithTracksJetEt = max(l1EndcapWithTracksJetEts)
        else:
            maxL1EndcapWithTracksJetEt = 0.

        if self._doEmu:
            l1EmuEndcapWithTracksJets = [jet for jet in event.l1EmuJets if ( abs(jet.eta) > 1.5 and abs(jet.eta) < 2.4 ) ]
            l1EmuEndcapWithTracksJetEts = [EndcapJet.et for EndcapJet in l1EmuEndcapWithTracksJets]
            nEmuEndcapWithTracksJets = len(l1EmuEndcapWithTracksJetEts)
            if nEmuEndcapWithTracksJets > 0:
                maxL1EmuEndcapWithTracksJetEt = max(l1EmuEndcapWithTracksJetEts)
            else:
                maxL1EmuEndcapWithTracksJetEt = 0.

        # Endcap without tracks Jets:
        l1EndcapNoTracksJets = [jet for jet in event.l1Jets if ( abs(jet.eta) > 2.4 and abs(jet.eta) < 3.0 ) ]
        l1EndcapNoTracksJetEts = [EndcapJet.et for EndcapJet in l1EndcapNoTracksJets]
        nEndcapNoTracksJets = len(l1EndcapNoTracksJets)
        if nEndcapNoTracksJets > 0:
            maxL1EndcapNoTracksJetEt = max(l1EndcapNoTracksJetEts)
        else:
            maxL1EndcapNoTracksJetEt = 0.

        if self._doEmu:
            l1EmuEndcapNoTracksJets = [jet for jet in event.l1EmuJets if ( abs(jet.eta) > 2.4 and abs(jet.eta) < 3.0 ) ]
            l1EmuEndcapNoTracksJetEts = [EndcapJet.et for EndcapJet in l1EmuEndcapNoTracksJets]
            nEmuEndcapNoTracksJets = len(l1EmuEndcapNoTracksJetEts)
            if nEmuEndcapNoTracksJets > 0:
                maxL1EmuEndcapNoTracksJetEt = max(l1EmuEndcapNoTracksJetEts)
            else:
                maxL1EmuEndcapNoTracksJetEt = 0.

        # Forward Jets
        l1HFJets = [jet for jet in event.l1Jets if abs(jet.eta) > 3.0]
        l1HFJetEts = [hfJet.et for hfJet in l1HFJets]
        nHFJets = len(l1HFJetEts)
        if nHFJets > 0:
            maxL1HFJetEt = max(l1HFJetEts)
        else:
            maxL1HFJetEt = 0.

        if self._doEmu:
            l1EmuHFJets = [jet for jet in event.l1EmuJets if abs(jet.eta) > 3.0]
            l1EmuHFJetEts = [hfJet.et for hfJet in l1EmuHFJets]
            nEmuHFJets = len(l1EmuHFJetEts)
            if nEmuHFJets > 0:
                maxL1EmuHFJetEt = max(l1EmuHFJetEts)
            else:
                maxL1EmuHFJetEt = 0.

        for name in self._jetTypes:
            if 'Emu' in name:
                if 'BE' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuBEJetEt)
                    getattr(self, name + '_rate_vs_pileup').fill(pileup,
                                                                 maxL1EmuBEJetEt)
                elif 'Barrel' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuBarrelJetEt)
                    getattr(self, name + '_rate_vs_pileup').fill(pileup,
                                                                 maxL1EmuBarrelJetEt)
                elif 'EndcapWithTracks' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuEndcapWithTracksJetEt)
                    getattr(self, name + '_rate_vs_pileup').fill(pileup,
                                                                 maxL1EmuEndcapWithTracksJetEt)
                elif 'EndcapNoTracks' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuEndcapNoTracksJetEt)
                    getattr(self, name + '_rate_vs_pileup').fill(pileup,
                                                                 maxL1EmuEndcapNoTracksJetEt)
                elif 'HF' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuHFJetEt)
                    getattr(self, name + '_rate_vs_pileup').fill(pileup,
                                                                 maxL1EmuHFJetEt)
                else:
                    getattr(self, name + '_rates').fill(pileup, maxL1EmuJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1EmuJetEt)
            else:
                if 'BE' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1BEJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1BEJetEt)
                elif 'Barrel' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1BarrelJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1BarrelJetEt)
                elif 'EndcapWithTracks' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EndcapWithTracksJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1EndcapWithTracksJetEt)
                elif 'EndcapNoTracks' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1EndcapNoTracksJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1EndcapNoTracksJetEt)
                elif 'HF' in name:
                    getattr(self, name + '_rates').fill(pileup, maxL1HFJetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1HFJetEt)
                else:
                    getattr(self, name + '_rates').fill(pileup, maxL1JetEt)
                    getattr(self,
                            name + '_rate_vs_pileup').fill(pileup, maxL1JetEt)

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
        # TODO: implement this in BaseAnalyzer
        # make_plots -> make_plots(plot_func)

        # Get EMU thresholds for each HW threshold.

        if 'thresholds' not in self.params:
            print(
                'Error: Please specify thresholds in the config .yaml in dictionary format')

        # print hw vs emu for rates and rate vs pileup
        for histo_name in self._sumTypes + self._jetTypes:
            plotter = getattr(self, histo_name + '_rates')
            if self._doEmu:
                if "_Emu" in histo_name:
                    continue
                emu_plotter = getattr(self, histo_name + '_Emu_rates')
                plotter.overlay([emu_plotter])
            if self._doComp:
                plotter.overlay([getattr(other_analyzer, histo_name + '_rates')
                                for other_analyzer in other_analyzers])

            plotter = getattr(self, histo_name + '_rate_vs_pileup')
            if self._doEmu:
                emu_plotter = getattr(self, histo_name + "_Emu" + '_rate_vs_pileup')
                plotter.overlay([emu_plotter])
            if self._doComp:
                plotter.overlay([getattr(other_analyzer, histo_name + '_rate_vs_pileup')
                                for other_analyzer in other_analyzers])

        # calculate cumulative histograms
        for plot in self.all_plots:
            if 'rate_vs_pileup' not in plot.filename_format:
                hist = plot.plots.get_bin_contents([bn.Base.everything])
                hist = cumulative_hist(hist)
                setattr(self, plot.online_name, hist)
                if not self._doEmu:
                    plot.draw()

        print('  thresholds:')

        if self.menuRates:
            for histo_name in self._sumTypes + self._jetTypes:
                h = getattr(self, histo_name)
                h = normalise_to_collision_rate(h)
                targetRate = self.menuRates.get(histo_name.replace('_Emu', ''))
                threshold = None
                closestRateDiff = 999999999
                threshold = None
                for i in range(h.nbins()):
                    if abs(h.get_bin_content(i) - targetRate) < closestRateDiff:
                        closestRateDiff = abs(h.get_bin_content(i) - targetRate)
                        threshold = h.get_bin_low_edge(i)
                outputLine = '    {0}:\t{1} KHz at threshold of \t\t{2}\tGeV'.format(
                    histo_name.ljust(15, ' '), targetRate, threshold)
                print(outputLine)
        print('\n')
        if self._doEmu:
            for histo_name in self._sumTypes + self._jetTypes:
                if "_Emu" in histo_name:
                    continue
                h = getattr(self, histo_name)
                h_emu = getattr(self, histo_name + "_Emu")
                thresholds = self.thresholds.get(histo_name)
                emu_thresholds = []
                for thresh in thresholds:
                    rate_delta = []
                    hw_rate = h.get_bin_content(thresh)
                    for i in range(h.nbins()):
                        emu_rate = h_emu.get_bin_content(i)
                        if hw_rate == 0. or emu_rate == 0.:
                            rate_delta.append(40000000.)
                        else:
                            rate_delta.append(abs(hw_rate - emu_rate))
                    emu_thresholds.append(rate_delta.index(min(rate_delta)))
                outputlineA = '    {0}: {1}'.format(histo_name, thresholds)
                outputlineB = '    {0}: {1}'.format(histo_name + '_Emu', emu_thresholds)
                outputline = outputlineA + '\n' + outputlineB

                print(outputline)

        '''
        for histo_name in object_types:
            h = getattr(self, histo_name)
            plot(h, histo_name, self.output_folder)
        '''
        return True

    def finalize(self):
        self.__print_histogram_statistics()
        return True

    def __print_histogram_statistics(self):
        all_stats = {}
        for plot in self.all_plots:
            # different hist collection will have different stats formats!
            collection_type = type(plot).__name__
            if collection_type not in all_stats:
                all_stats[collection_type] = []

            summary_label = 'PU'
            summary_bins = self.puBins
            if collection_type == 'RatesPlot':
                summary_label = 'bin'
                for var, thresholds in self.thresholds.items():
                    if "L1 " + var in plot.online_title:
                        summary_bins = thresholds
            all_stats[collection_type] += [plot.get_stats(summary_label=summary_label, summary_bins=summary_bins)]

        for collection_type in all_stats:
            df = pd.concat(all_stats[collection_type])  # , sort=False)
            df.sort_values(by=['identifier'], inplace=True)
            df.fillna('------', inplace=True)
            print('Histogram collection:', collection_type)
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            df_output = os.path.join(self.output_folder, '{}_histogram_stats.csv'.format(collection_type))
            df.to_csv(df_output)
            print('Saved histogram statistics to', df_output)


def plot(hist, name, output_folder):
    pu = ''
    if '_pu' in name:
        pu = name.split('_')[-1]
        name = name.replace('_' + pu, '')
    file_name = 'rates_{name}'.format(name=name)
    if 'nVertex' in name:
        file_name = 'nVertex'
    if pu:
        file_name += '_' + pu
    canvas_name = file_name.replace('SingleMu', 'Rates')
    c = ROOT.TCanvas(canvas_name)
    if 'nVertex' not in name:
        c.SetLogy()
    hist.set_y_title('Rate (Hz)')
    hist.set_x_title(name)
    hist.Draw()
    c.SaveAs(os.path.join(output_folder, file_name + '.pdf'))
