from os.path import join
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def saveallforms(figdirs, fignm, figh=None, fmts=("png", "pdf")):
    """Save all forms of a figure in an array of directories."""
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx), bbox_inches='tight')

