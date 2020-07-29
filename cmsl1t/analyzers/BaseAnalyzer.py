import os
from rootpy.io import root_open
from rootpy import ROOTError
import logging
logger = logging.getLogger(__name__)


class BaseAnalyzer(object):
    DEFAULT_OUTPUT_FORMAT = 'pdf'
    """
    A Base class to be used by the various analyzers
    """

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.output_folder = kwargs.pop('output_folder')
        self.plots_folder = kwargs.pop('plots_folder')
        self.all_plots = []
        self.file_format = kwargs.pop('file_format')
        self.__params = kwargs
        self._doComp = False

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
            os.makedirs(self.plots_folder+'/DetailledPlots/')
    @property
    def params(self):
        return self.__params

    def prepare_for_events(self, reader):
        """
        Can be overloaded in the derived class.
        Called once, after the input trees have been prepared.

        returns:
          Should return True if everything was prepared ok.
          If anything else is returned, processing will stop
        """
        return True

    def process_event(self, entry, event):
        """Should not really be overloaded in the derived class"""
        return self.fill_histograms(entry, event)

    def fill_histograms(self, entry, event):
        """
        Has to be overloaded by users code.

        Called once per input event in the tuples.

        parameters:
         - entry -- the index of the entry being read in
         - event -- the event data read in from the tuples

        returns:
          Should return True if histograms were filled without problem.
          If anything else is returned, processing of the trees will stop
        """
        raise NotImplementedError("fill_histograms needs to be implemented")
        return True

    def reload_histograms(self, input_filename, do_comparison=False):
        """
        Read back histograms from the given root file.
        May need to append histograms

        returns:
          Should return True if histograms were written without problem.
          If anything else is returned, processing of the trees will stop
        nb:
             If comparing multiple files, legend title "comp_title" is
             provided with the filename, and passed to from_root
         """
        results = []
        if do_comparison:
            self._doComp = True
            comp_title = input_filename.split(':')[0]
            input_filename = input_filename.split(':')[1]
        with root_open(input_filename, "r") as input_file:
            for hist in self.all_plots:
                indir = input_file.GetDirectory(hist.directory_name)
                results.append(hist.from_root(indir, comp_title))
        ok = all(results)
        return ok

    def write_histograms(self):
        """
        Called after all events have been read, so that histograms can be
        written to file).  Note that plots should not be drawn here, since this
        method will not be called if we are running off previously filled
        histograms.

        returns:
          Should return True if histograms were written without problem.
          If anything else is returned, processing of the trees will stop
        """
        results = []
        outname = self.get_histogram_filename()
        try:
            with root_open(outname, "new") as outfile:
                logger.info("Saving histograms to: " + outname)
                for hist in self.all_plots:
                    outdir = outfile.mkdir(hist.directory_name)
                    results.append(hist.to_root(outdir))
        except ROOTError:
            # Root file already exists, not handled by root_open
            pass
        return all(results)

    def make_plots(self, other_analyzers=None):
        """
        Called after all events have been read to convert histograms to actual
        plots Might be called on existing files of histograms (ie. without
        reading tuples in again)

        returns:
          Should return True if plots were produced without problem.
          If anything else is returned, processing of the trees will stop
        """
        for plot in self.all_plots:
            plot.draw()
        return True

    def finalize(self):
        """
        Can be overloaded in the derived class if needed.

        Called at the very end of the code to tidy things up before the
        programs quits
        """
        return True

    def register_plotter(self, plotter):
        """
        Register a plotter with this analyzer, and set up it's outputs
        """
        plotter.set_plot_output_cfg(self.plots_folder, self.file_format)
        self.all_plots.append(plotter)

    _hist_file_format = "{analyzer}_histograms.root"

    def get_histogram_filename(self):
        output_file = self._hist_file_format.format(analyzer=self.name)
        return os.path.join(self.output_folder, output_file)

    def might_contain_histograms(self, filename):
        this_file = self._hist_file_format.format(analyzer=self.name)
        base = os.path.basename(filename)
        return base == this_file
